from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from sklearn.metrics import roc_auc_score
import torch
import wandb
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomTrainer:
    def __init__(
            self,
            batch_size,
            optimizer=None,
            scheduler=None,
            loss=None,
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.batch_size = batch_size
        self._cached_model = None
        self.skip_seq_len = None

    def _prepare_dataloader(
            self,
            dataset,
            shuffle,
    ):
        """
        additional dataloader transforming(used only for group_mode == True, because of tuple batches)
        """
        if hasattr(dataset, 'group_mode') and dataset.group_mode:

            batch_sampler = IndexBatchSampler(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle
            )
            dataloader = DataLoader(
                dataset,
                batch_size=None,
                sampler=batch_sampler,
            )
        else:
            dataloader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=shuffle
            )

        return dataloader

    def _pass_batch(
            self,
            input,
            target,
            return_pred_target: bool = False,
            only_single: bool = False,
    ):
        # todo. it's a very dumb solution for solving data allocation in cuda
        if self.skip_seq_len is not None:
            max_len = 0
            if isinstance(input, list) and len(input) == 0:
                return None

            if isinstance(input, list) and len(input) > 0:
                max_len = torch.max(torch.tensor([i.shape[-1] for i in input]))
            elif isinstance(input, torch.Tensor):
                max_len = input.shape[-1]

            if max_len > self.skip_seq_len:
                return None

        # if target is np array -> target is a list of tokens with different length -> pad+tensor
        # the very same thing means, that we want to process token though model without any pooler
        at_mask = 1
        pool = True
        if isinstance(target, np.ndarray):
            pool = False
            target, at_mask = self._cached_model.pad_list_to_tensor(target, padding_value=0)
            # because to multiply (b,n,c) on mask we should expand it from (b,n) to (b,n,1) for broadcasting
            at_mask = at_mask.unsqueeze(-1).to(device)

        # multitarget split
        if only_single:
            multitarget_mask = torch.tensor([False] * len(target))  # base case
        else:
            multitarget_mask = torch.sum(target, dim=-1) != 1

            # if we have targets as tokens, than recreate mask to batch idx
            if multitarget_mask.dim() > 1:
                multitarget_mask = torch.any(multitarget_mask, dim=1)

        prediction = self._cached_model(input, multitarget_mask, pool) * at_mask
        # multitarget rearangment
        target = torch.concat([target[multitarget_mask], target[~multitarget_mask]]).squeeze(1)
        target = target.to(device)

        grad_output = self.loss(prediction, target)

        if return_pred_target:
            return grad_output, prediction.cpu().detach(), target.cpu().detach()
        else:
            return grad_output

    def _pass_batch_nano_vram(
            self,
            input,
            target,
            return_pred_target: bool = False,
            only_single: bool = False,
            micro_batch_size: int = 1,
    ):
        gradients_magn = []
        predictions = []
        targets = []
        gradient_magn = 0
        procesed_micro_batches = 0
        for k in range(0, len(target), micro_batch_size):
            i = input[k: k + micro_batch_size]
            t = target[k: k + micro_batch_size]
            res = self._pass_batch(
                i,
                t,
                return_pred_target,
                only_single
            )
            if res is None:
                continue
            procesed_micro_batches += 1
            if return_pred_target:  # TODO IT'S ONLY FOR EVAL
                g, p, t = res
                gradients_magn.append(g.item())
                predictions.append(p.cpu().detach())
                targets.append(t.cpu().detach())
            else:  # TODO IT'S ONLY FOR TRAIN
                res = res * micro_batch_size / len(target)
                res.backward()
                gradient_magn += res.item()

        if procesed_micro_batches == 0:
            return None
        if gradient_magn != 0:
            return gradient_magn, procesed_micro_batches
        else:
            return gradients_magn, predictions, targets, procesed_micro_batches

    def train(
            self,
            model,
            train_idx=None,
            eval_idx=None,
            ds_class=None,
            train_ds=None,
            eval_ds=None,
            epochs: int = None,
            steps: int = None,
            # if steps are provided, than eval on each eval_freqth step, if epochs are provided eval on each (float part of epoch (e.g. each 0.3 epoch))
            eval_freq: float = None,
            # number of batches to process beefore updating
            optim_freq: int = 1,
            # the same as in eval_freq
            save_freq: float = None,
            skip_seq_len: int = None,
            micro_batch_size: int = 1,
    ):

        train_dataloader = self._prepare_dataloader(train_ds, shuffle=True)
        eval_dataloader = self._prepare_dataloader(eval_ds, shuffle=False)

        self.skip_seq_len = skip_seq_len
        self._cached_model = model
        if (epochs is None) and (steps is None):
            raise ValueError("epochs and steps can not be Nones simultaneously")

        if epochs is not None:
            steps = epochs * len(train_dataloader)
            eval_freq = int(eval_freq * len(train_dataloader))
            save_freq = int(save_freq * len(train_dataloader))

        cum_loss = 0
        batches_processed = 0
        train_iter = iter(train_dataloader)
        progress_bar = tqdm(range(steps), desc="training")
        self._cached_model
        self._cached_model.train()

        for i in progress_bar:
            # continuously pulling from an iterator
            try:
                input, target = next(train_iter)

            except StopIteration:
                # reset iterator if it runs out before 'steps' is reached
                train_iter = iter(train_dataloader)
                input, target = next(train_iter)

            if micro_batch_size == 0:
                output = self._pass_batch(
                    input,
                    target,
                    return_pred_target=False,
                    only_single=False,  # todo set as parameter
                )
                if output is None:
                    del input, target
                    continue

                batches_processed += procesed_micro_batches
                cum_loss += output.item()
                output.backward()

            else:
                output, procesed_micro_batches = self._pass_batch_nano_vram(
                    input,
                    target,
                    return_pred_target=False,
                    only_single=False,
                    micro_batch_size=micro_batch_size,
                )
                if output is None:
                    del input, target
                    continue

                batches_processed += procesed_micro_batches
                cum_loss += output  # .item()

            if (i + 1) % optim_freq == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            # (i+1)%freq <- e.g. i = 1 (2nd iter), freq = 2 (each 2 steps) -\_( :) )_/-
            if (i != 0) and ((i + 1) % eval_freq == 0):
                # eval loss per sample
                eval_loss, roc_score = self.eval(eval_dataloader)
                eval_loss /= len(eval_dataloader)
                # train loss per sample
                if micro_batch_size == 0:
                    train_loss = cum_loss / (batches_processed * self.batch_size)
                else:
                    train_loss = cum_loss / (batches_processed * micro_batch_size)

                log_dict = {
                    "train/loss": train_loss,
                    "eval/loss": eval_loss,
                    "eval/roc_auc": roc_score,
                    "step": i,
                }

                if self.scheduler:
                    log_dict["train/lr"] = self.optimizer.param_groups[0]['lr']

                wandb.log(log_dict)
                # print(f"step: {i}\ntrain_loss: {train_loss}\neval_loss: {eval_loss}\neval roc score: {roc_score}")
                cum_loss = 0
                batches_processed = 0
                self._cached_model.train()

            if (i + 1) % save_freq == 0:
                torch.save(model.state_dict(), f'dmodel_{i}.pt')

        return self._cached_model

    @torch.no_grad()
    def predict(
            self,
            model,
            dataset,
            multitarget: bool = False
    ):
        """
        create predictions for each filename and store in dataset.predictions as pd.series
        """
        dataloader = self._prepare_dataloader(dataset, shuffle=False)
        model.eval()
        for input, _, file_path in dataloader:
            # input is a list of tensor
            multitarget_mask = torch.tensor([multitarget] * len(input))
            output = model(input, multitarget_mask, model.classifier.seq_mode)

            # if we run model in predicting only 1 vector for file, it remains to be a torch tensor (b,c)
            # in that case we create a [np.array(c), ...(n-1)], to save predictions as a pd.series
            if isinstance(output, torch.Tensor):
                # for np.version > 2.1 should use simple .unstack()
                output = np.split(output.cpu().detach().numpy(), output.shape[0], axis=0)

            pretty_output = pd.Series(data=output, index=file_path)
            dataset.update_pseudo_labels(pretty_output)

    @torch.no_grad()  # todo actually to inspect the magnitude of update isn't a bad idea, (remember TS)
    def eval(
            self,
            eval_dataloader,
    ):
        """
        evaluation implemented only in kaggle way (1 class)
        """
        # todo handle multitarget target with roc auc
        self._cached_model.eval()

        # todo we can rewrite it invo model.eval() method
        return_NoF_buf = self._cached_model.return_NoF
        if return_NoF_buf:
            self._cached_model.return_NoF = False

        cum_loss = 0
        y_true = []
        y_pred = []
        for input, target in eval_dataloader:

            result = self._pass_batch_nano_vram(
                input,
                target,
                return_pred_target=True,
                only_single=False,
                micro_batch_size=1,
            )
            if result is None:
                del input, target
                continue

            output, prediction, target, procesed_batches = result
            y_true.extend(target)
            y_pred.extend(prediction)

            cum_loss += torch.sum(torch.tensor(output)) / procesed_batches

        # turning back on
        self._cached_model.return_NoF = return_NoF_buf

        # roc calc
        # datatypes transition
        # also drop off the last digit, because the last digit is NoF
        y_true = torch.concat(y_true, dim=0).cpu().detach().numpy()
        y_pred = torch.concat(y_pred, dim=0).cpu().detach().numpy()
        # roc mask, to include only presented classes
        class_sums = y_true.sum(axis=0) > 0

        y_true = y_true[:, class_sums]
        y_pred = y_pred[:, class_sums]

        roc = roc_auc_score(
            y_true,
            y_pred,
            average='macro',
        )
        return cum_loss, roc