import numpy as np
import torch 
import copy

def numpify(func):
    """Wrapper so that the augmentation function always works on a numpy
    array, but if the input `imgs` is a torch tensor, a torch tensor will be
    returned. Assumes first input and first output of the function is the
    images array/tensor, and only operates on that."""
    def numpified_aug(imgs, *args, **kwargs):
        _numpify = isinstance(imgs, torch.Tensor)
        if _numpify:
            imgs = imgs.numpy()
        ret = func(imgs, *args, **kwargs)
        if _numpify:
            if isinstance(ret, tuple):
                # Assume first is the augmented images.
                ret = (torch.from_numpy(ret[0]), *ret[1:])
            else:
                ret = torch.from_numpy(ret)
        return ret
    return numpified_aug

@numpify
def random_shift(imgs, pad=1, prob=1.):
    t = b = c = 1
    shape_len = len(imgs.shape)
    if shape_len == 2:  # Could also make all this logic into a wrapper.
        h, w = imgs.shape
    elif shape_len == 3:
        c, h, w = imgs.shape
    elif shape_len == 4:
        b, c, h, w = imgs.shape
    elif shape_len == 5:
        t, b, c, h, w = imgs.shape  # Apply same crop to all T
        imgs = imgs.transpose(1, 0, 2, 3, 4)
        _c = c
        c = t * c
        # imgs = imgs.reshape(b, t * c, h, w)
    imgs = imgs.reshape(b, c, h, w)

    crop_h = h
    crop_w = w

    padded = np.pad(
        imgs,
        pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="edge",
    )
    b, c, h, w = padded.shape

    h_max = h - crop_h + 1
    w_max = w - crop_w + 1
    h1s = np.random.randint(0, h_max, b)
    w1s = np.random.randint(0, w_max, b)
    if prob < 1.:
        which_no_crop = np.random.rand(b) > prob
        h1s[which_no_crop] = pad
        w1s[which_no_crop] = pad

    shifted = np.zeros_like(imgs)
    for i, (pad_img, h1, w1) in enumerate(zip(padded, h1s, w1s)):
        shifted[i] = pad_img[:, h1:h1 + crop_h, w1:w1 + crop_w]

    if shape_len == 2:
        shifted = shifted.reshape(crop_h, crop_w)
    elif shape_len == 3:
        shifted = shifted.reshape(c, crop_h, crop_w)
    elif shape_len == 5:
        shifted = shifted.reshape(b, t, _c, crop_h, crop_w)
        shifted = shifted.transpose(1, 0, 2, 3, 4)

    return shifted

class ContrastModel(torch.nn.Module):

    def __init__(self, latent_size:int, anchor_hidden_size:int, tau:float=1.0):
        super().__init__()
        self.projector = torch.nn.Linear(latent_size, 256)
        # no gradient to projector target
        self.projector_target = copy.deepcopy(self.projector)
        for param in self.projector_target.parameters():
            param.requires_grad = False

        latent_size = 256

        self.anchor_mlp = torch.nn.Sequential(
            torch.nn.Linear(latent_size, anchor_hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(anchor_hidden_size, latent_size),
        )
        self.W = torch.nn.Linear(latent_size, latent_size, bias=False)
        self.c_e_loss = torch.nn.CrossEntropyLoss()
        self.predictor = torch.nn.Linear(latent_size, latent_size)
        
    def forward(self, anchor, positive):
        anchor = self.projector(anchor)
        positive = self.projector_target(positive)
        if self.anchor_mlp is not None:
            anchor = anchor + self.anchor_mlp(anchor)  # skip probably helps
        pred = self.W(anchor)
        logits = torch.matmul(pred, positive.T)
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]  # normalize
        return logits, anchor, positive

    def calculate_loss(self, anchor, positive, B, T, extra):
        labels = torch.arange(anchor.shape[0],
            dtype=torch.long, device=anchor.device)

        logits, anchor, positive = self(anchor, positive)
        ul_loss = self.c_e_loss(logits, labels)
        
        anchor = anchor.view(B, T, -1)
        positive = positive.view(B, T, -1)

        anchor_left, anchor_right = anchor[:, :-1], anchor[:, 1:]
        positive_left, positive_right = positive[:, :-1], positive[:, 1:]

        anchor_left = anchor_left.reshape(B * (T - 1), -1)
        anchor_right = anchor_right.reshape(B * (T - 1), -1)
        positive_left = positive_left.reshape(B * (T - 1), -1)
        positive_right = positive_right.reshape(B * (T - 1), -1)

        anchor_right_pred = self.predictor(anchor_left)
        positive_right_pred = self.predictor(positive_left)

        extra_loss = torch.nn.functional.mse_loss(anchor_right_pred, anchor_right) + torch.nn.functional.mse_loss(positive_right_pred, positive_right)

        if extra:
            return {"atc_loss": ul_loss, "extra_loss": extra_loss}
        else: 
            return {"atc_loss": ul_loss}

    def update_momentum(self, m):
        for target_param, param in zip(self.projector_target.parameters(), self.projector.parameters()):
            target_param.data = m * param.data + (1 - m) * target_param.data



def update_state_dict(model, state_dict, tau=1, strip_ddp=True):
    """Update the state dict of ``model`` using the input ``state_dict``, which
    must match format.  ``tau==1`` applies hard update, copying the values, ``0<tau<1``
    applies soft update: ``tau * new + (1 - tau) * old``.
    """
    if strip_ddp:
        state_dict = strip_ddp_state_dict(state_dict)
    if tau == 1:
        model.load_state_dict(state_dict)
    elif tau > 0:
        update_sd = {k: tau * state_dict[k] + (1 - tau) * v
            for k, v in model.state_dict().items()}
        model.load_state_dict(update_sd)

def strip_ddp_state_dict(state_dict):
    """ Workaround the fact that DistributedDataParallel prepends 'module.' to
    every key, but the sampler models will not be wrapped in
    DistributedDataParallel. (Solution from PyTorch forums.)"""
    clean_state_dict = type(state_dict)()
    for k, v in state_dict.items():
        key = k[7:] if k[:7] == "module." else k
        clean_state_dict[key] = v
    return clean_state_dict
