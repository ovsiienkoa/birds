from torch import nn
import torch
import timm
import torchaudio.transforms as transforms
import numpy as np

from deepseek_mla.src import MultiHeadLatentAttention #proper installation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ETransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_size,
        num_head,
        d_embed,
        d_c,
        d_c1,
        d_rotate,
    ):
        super().__init__()
        self.rms_n = nn.RMSNorm(embedding_size)
        self.att = MultiHeadLatentAttention(
            d_model = embedding_size, #after attention embedding_size
            num_head = num_head,
            d_embed = d_embed, #input to attention embedding_size
            d_c = d_c,
            d_c1 = d_c1,
            d_rotate = d_rotate,
            dropout=0.1,
            bias=True,
        )
        self.liner = nn.Linear(embedding_size, embedding_size)
        self.lin_act = nn.SiLU()

    def forward(self, x, att_mask):
        res = x
        x = self.rms_n(x)
        x = self.att(sequence = x, att_mask = att_mask)
        x += res
        res = x
        x = self.rms_n(x)
        x = self.liner(x)
        x = self.lin_act(x)
        x += res
        return x


class TokenEncoder(nn.Module):
    def __init__(
            self,
            embedding_size,
            class_num,
            num_head,
            d_c,
            d_c1,
            d_rotate,
            pool_type: str = 'cls',
    ):
        super().__init__()
        # self.vembed2embed = nn.Linear(vembedding_size, embedding_size)
        self.output_size = class_num + 1
        self.embedding_size = embedding_size
        self.num_head = num_head
        self.embed2embed = ETransformerBlock(
            self.embedding_size,
            num_head,
            self.embedding_size,
            d_c,
            d_c1,
            d_rotate,
        )
        self.embed2class = nn.Linear(embedding_size, self.output_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_size))
        self.pool_type = pool_type

    def forward(
            self,
            x,
            return_NoF,
            attention_mask: torch.Tensor = None,
            pool: bool = True,
    ):

        batch_size = x.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls, x), dim=1)  # B, T, E -> B, T+1, E
        # todo attention mask expand
        attention_mask = torch.cat(
            (torch.ones(attention_mask.shape[0]).unsqueeze(-1), attention_mask),
            dim=-1
        )  # B, T -> B, T+1 {1, 0}

        q_mask = attention_mask.unsqueeze(2)
        k_mask = attention_mask.unsqueeze(1)
        attention_matrix = q_mask * k_mask  # (B, T+1, T+1)
        attention_matrix -= 1  # {0 ,-1}
        attention_matrix *= 1e9  # {0, -inf}
        attention_matrix = attention_matrix.unsqueeze(1).expand(-1, self.num_head, -1, -1).to(
            device)  # (B, H, T+1, T+1)
        output = self.embed2embed(x,
                                  att_mask=attention_matrix)  # B, T+1, E -> B, T+1, E #todo probably bolleans to int in att_mask
        # output = x #for simplicity in test setup

        if pool:
            # or in model cls_token pool
            if self.pool_type == 'cls':
                output = output[:, 0, :]  # (B, 1, E)

            # or token_avg pool
            elif self.pool_type == 'mean':

                output = output[:, 1:, :].to(device)
                attention_mask = attention_mask[:, 1:].to(device)
                last_indx = torch.sum(attention_mask, dim=1).int().unsqueeze(-1).to(device)  # (B, 1)

                # handle empty batch
                if output.shape[0] == 0:
                    output = torch.empty(0, self.embedding_size, dtype=output.dtype, device=output.device)
                else:
                    # because 'output' contains padded tokens (with zeros) calculated stats will also include them
                    # to play it safe, I want to multiply 'output' on 'att_mask'
                    output = (output * attention_mask.unsqueeze(-1)).sum(dim=1) / last_indx  # B, T, E -> B, E
            else:
                raise ValueError(f"pool_type {self.pool_type} isn't implemented yet")
        else:
            output = output[:, 1:, :]

        output = self.embed2class(output)  # B, ?, E -> B, ?, C+1

        if not return_NoF:
            # last digit is NoF
            output = output[..., :-1]

        return output

class Encoder(nn.Module):
    def __init__(self, backbone_name: str, original_weights: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=original_weights,
            num_classes=0,
        )
        # init data-transform functions
        self.stft_transform = None
        self.db_transform = None
        self.mel_transform = None
        self.pooler = None

    def set_audiopreprocessing(self, cfg):
        self.stft_transform = transforms.Spectrogram(
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            power=1.0,
        )
        self.db_transform = transforms.AmplitudeToDB(
            stype="magnitude",
            top_db=cfg.top_db
        )
        self.mel_transform = transforms.MelScale(
            sample_rate=cfg.sr,
            n_stft=cfg.n_fft // 2 + 1,
            n_mels=cfg.n_mels,
        )
        self.pooler = nn.AdaptiveMaxPool1d(cfg.n_mels)

    def stripe_w_overlap(
            self,
            ar: torch.Tensor,
            stripe: int,
            overlap: int,
            pad_value: float,
    ):
        """
        splits tensor into overlapping chunks along time dim
        the last token is dropped
        if the time < token_width => pad with constant value of pad_value
        returns (B(depends on input), C, H, W, tokens)
        """
        step = stripe - overlap
        time_len = ar.shape[-1]  # time

        # if time < stripe => pad with zeros, because we can't stack None values
        if time_len < stripe:
            ar = torch.nn.functional.pad(
                input=ar,
                pad=(0, stripe - time_len),
                mode='constant',
                value=pad_value

            )
            num_steps = 1
        # if time < stripe+step means that there is possible only 1 step
        elif time_len < stripe + step:
            num_steps = 1
        # number of full chunks
        else:
            num_steps = (time_len - stripe) // step

        try:
            striped_tensor = torch.stack(
                [
                    ar[:, ..., i * step: i * step + stripe]
                    for i in range(num_steps)
                ]
            )  # (T, ...)
        except:
            raise RuntimeError(
                f"error in stripes. here is the tensor shape: {ar.shape} ; the stripe: {stripe} ; and overlap: {overlap} ; and steps: {num_steps} ")

        striped_tensor = torch.moveaxis(striped_tensor, 0, -1).contiguous()  # (T, ...) -> (..., T)

        return striped_tensor

    def forward(self, x, **kwargs):
        # (B*T, C, H, W) -> (B*T, C_backbone)

        # ==prerpcoessing raw waves into images
        # (B*T, C, H, W) -> (B, C_spects, C_mel, W_stft)

        stft = self.stft_transform(x)
        stft = stft[..., :320]  # todo idk why, but I started to receive 321 windows after moving striping on raw waves
        # conver to dB spectrs
        mel_out_stft = self.db_transform(self.mel_transform(stft))
        mel_out_stft_2 = self.db_transform(self.mel_transform(stft ** 2))
        stft = self.db_transform(stft)

        # normalize stft to mel size
        mega_B, H, W, C_stft, W_stft = stft.shape
        stft = torch.transpose(stft, -1, -2)  # mega_B, H, W, C_stft, W_stft -> mega_B, H, W, W_stft, C_stft

        # flatten B, H, W, T into a single *B* dimension
        stft = stft.reshape(-1, 1, C_stft)  # (B,H,W,W_s,C) -> (*B*, 1, C)
        pooled_lin = self.pooler(stft)  # (*B*, 1, C_stft) -> (*B*, 1, C_mel)
        pooled_lin = pooled_lin.reshape(mega_B, H, W, W_stft, -1)  # (mega_B, H, W, W_stft, C_mel)

        x = torch.stack((mel_out_stft, mel_out_stft_2, pooled_lin), dim=1)  # B, C_spects, H, W, C_mel, W_stft
        x = x.squeeze(2).squeeze(2)  # B, C_spects, C_mel, W_stft
        # ==prerpcoessing raw waves into images

        return self.backbone(x)  # (B*T, C, H, W) -> (B*T, C)


class Classifier(nn.Module):
    def __init__(
            self,
            seq_mode=True,  # to process instance wise or sequence wise
            single_head: nn.Module = None,
            multi_head: nn.Module = None,
            single_activation: nn.Module = None,
            multi_activation: nn.Module = None,

    ):
        super().__init__()
        self.seq_mode = seq_mode
        self.single_target_model = single_head
        self.multi_target_model = multi_head
        # activation fn in this class instead of token encoder, if I want to thrain only 1 SED head
        self.single_activation = single_activation
        self.multi_activation = multi_activation

    def forward(
            self,
            x,
            multitarget_mask=None,
            attention_mask: torch.Tensor = None,
            return_NoF: bool = False,
            pool: bool = True,
    ):
        """
        Pass padded vision embeddings in classifier with implementing additional logic:
        * multi target model split(multitarget_mask)
        * removes NoF token (return_NoF = False)
        * returns prediction token wise (pool = False)

        the classifier object's attribute seq_mode determines, whether to return list of token-wise predictions
        or padded tensor of token-wise predictions
        Args:
            x: features from vision encoder.
            multitarget_mask: list of booleans, where True means to use multitarget model.
            return_NoF: whether to return NoF token.
            pool: whether to return output sequence wise or token wise.
        """

        # B, T_max, Channels -> B, empty/T_max, Class_digit+NoF
        # we have rotary embeddings, so we can't pass empty batches anymore
        if torch.all(multitarget_mask) == True:
            output = self.multi_target_model(x, return_NoF, attention_mask, pool)
            output = self.multi_activation(output)
        elif torch.any(multitarget_mask) == True:
            multi_output = self.multi_target_model(x[multitarget_mask], return_NoF, attention_mask[multitarget_mask],
                                                   pool)
            multi_output = self.multi_activation(multi_output)
            single_output = self.single_target_model(x[~multitarget_mask], return_NoF,
                                                     attention_mask[~multitarget_mask], pool)
            single_output = self.single_activation(single_output)
            output = torch.cat([multi_output, single_output], dim=0)
        else:
            output = self.single_target_model(x, return_NoF, attention_mask, pool)
            output = self.single_activation(output)

        if not self.seq_mode:

            masked_output = []
            last_indx = torch.sum(attention_mask, dim=1).int()

            # list of token predictions for each instance
            for i, out in enumerate(output):
                masked_output.append(out[:last_indx[i]].cpu().detach())

            output = masked_output

        return output


class CLEFModel(nn.Module):
    def __init__(
            self,
            encoder=None,
            classifier=None,
            padding_value=0,
            return_NoF=False,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.padding_value = padding_value
        self.return_NoF = return_NoF

    def pad_list_to_tensor(self, tensor_list: list, padding_value: float = None):
        """
        takes a list of embeddings [(T_i, C), ...] and creates out of them 1 padded tensor
        """
        if padding_value is None:
            padding_value = self.padding_value
        # in batch maximum for padding (in tokens)
        max_len = np.max([s.shape[0] for s in tensor_list])

        batch_list = []
        attention_mask_list = []
        for tensor in tensor_list:
            # create attention mask, to prevent usage of padded values in the future
            attention_mask = torch.ones(tensor.shape[0])
            attention_mask = torch.nn.functional.pad(
                attention_mask,
                (0, max_len - tensor.shape[0]),  # token wise padding
                mode='constant',
                value=0
            )

            attention_mask_list.append(attention_mask)

            tensor = torch.nn.functional.pad(
                tensor,
                (0, 0, 0, max_len - tensor.shape[0]),  # token wise padding
                mode='constant',
                value=padding_value  # equivavelnt of zero
            )

            batch_list.append(tensor)

        batch_tensor = torch.stack(batch_list, dim=0)  # B, max(T_b), C
        attention_mask = torch.stack(attention_mask_list, dim=0)  # B, max(T_b)

        return batch_tensor, attention_mask

    def forward(
            self,
            x,
            multitarget_mask,
            pool,
    ):
        """
        Pass list of sequences into vision encoder and classifier models.

        seq-wise:
        Union list of tensor into huge one, to pass it through encoder,
        The received features are padded, to create once again 1 huge tensor for classifier
        """
        # list -> predict seq-wise
        if not isinstance(x, torch.Tensor):

            # huge tensor for feature encoding
            x_feature_tensor = torch.cat(x, dim=-1).permute(-1, 0, 1, 2).to(device)  # (T*B, C, H, W)
            try:
                x_feature_tensor = self.encoder(x_feature_tensor)  # (T*B, C)
            except:
                raise ValueError(f"Most likely cuda memory allocation, tensor shape: {x_feature_tensor.shape}")

            # split back into seqs
            prev_len = 0
            embed = []  # (B, T_b, C)
            for sub_x in x:
                cur_len = sub_x.shape[-1]
                embed.append(x_feature_tensor[prev_len: prev_len + cur_len])  # (T_b, C)
                prev_len = prev_len + cur_len

            # pad sequences
            embed, attention_mask = self.pad_list_to_tensor(embed)
        else:
            embed = self.encoder(x.to(device))
            attention_mask = torch.ones((embed.shape(0), embed.shape(1)))

        probs = self.classifier(
            x=embed,
            attention_mask=attention_mask,
            multitarget_mask=multitarget_mask,
            return_NoF=self.return_NoF,
            pool=pool
        )
        return probs