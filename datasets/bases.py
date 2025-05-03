from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import regex as re
import copy
from torchvision.transforms import ToTensor


'''
MLM (text)	MIM (image)
Input: Tokenized text	Input: Image patches
Mask: Random tokens	Mask: Random image patches
Output: Predict masked tokens	Output: Reconstruct masked patches
Model: Transformer (BERT)	Model: Vision Transformer (ViT)
'''
class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))

def patchify(img, patch_size):
    B,C,H,W = img.shape
    assert H % patch_size == 0 and W % patch_size == 0
    h,w = H//patch_size, W//patch_size

    patches = img.unfold(2,patch_size,patch_size).unfold(3,patch_size,patch_size)
    patches = patches.permute(0,2,3,1,4,5).reshape(B,-1,patch_size*patch_size*C)

    return patches

def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': tokens,
        }

        return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
            'mlm_ids': mlm_tokens,
            'mlm_labels': mlm_labels
        }

        return ret

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.15:
                    prob /= 0.15

                    # 80% randomly change token to mask token
                    if prob < 0.8:
                        tokens[i] = mask

                    # 10% randomly change token to random token
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)

                    # -> rest 10% randomly keep current token

                    # append current token to output (we will predict these later)
                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)

class ImageTextMaskMultimodalDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 patch_size=16,
                 mask_ratio=0.75,
                 text_length: int = 77,
                 truncate: bool = True,
                 enable_mim=True,
                 enable_mlm=True,
                 tokenizer=None):
        self.dataset = dataset
        self.transform = transform
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.text_length = text_length
        self.truncate = truncate
        self.enable_mim = enable_mim
        self.enable_mlm = enable_mlm

        self.tokenizer = tokenizer if tokenizer is not None else SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        #img = ToTensor()(read_image(img_path))/255.0
        # .float()/255.0
        img = read_image(img_path)
        #print(f"This is my img type: {type(img)}")
        

        if self.transform is not None:
            img = self.transform(img)  # [3, H, W]

        output = {
            'pids': pid,
            'image_ids': image_id,
        }

        # ====== MIM Part ======
        if self.enable_mim:
            patches = self._patchify(img)  # [N, patch_dim]
            masked_patches, mask, ids_restore = self._random_masking(patches, self.mask_ratio)

            output.update({
                'images': img,
                'patches': patches,
                'masked_patches': masked_patches,
                'mask': mask,
                'ids_restore': ids_restore
            })

        else:
            output['images'] = img

        # ====== MLM Part ======
        if self.enable_mlm:
            caption_tokens = tokenize(caption, tokenizer=self.tokenizer,
                                      text_length=self.text_length,
                                      truncate=self.truncate)

            mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(caption_tokens.cpu().numpy())
            output.update({
                'caption_ids': caption_tokens,
                'mlm_ids': mlm_tokens,
                'mlm_labels': mlm_labels
            })

        return output

    def _patchify(self, img):
        C, H, W = img.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        h, w = H // self.patch_size, W // self.patch_size

        patches = img.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, C * self.patch_size * self.patch_size)
        return patches

    def _random_masking(self, patches, mask_ratio):
        N, D = patches.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(N)
        ids_shuffle = torch.argsort(noise, dim=0)
        ids_restore = torch.argsort(ids_shuffle, dim=0)

        ids_keep = ids_shuffle[:len_keep]
        x_masked = patches[ids_keep]

        mask = torch.ones(N)
        mask[:len_keep] = 0
        mask = mask[ids_restore]

        return x_masked, mask, ids_restore

    def _build_random_masked_tokens_and_labels(self, tokens):
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder) - 3))  # 1 ~ 49405

        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15
                    if prob < 0.8:
                        tokens[i] = mask
                    elif prob < 0.9:
                        tokens[i] = random.choice(token_range)
                    labels.append(token)
                else:
                    labels.append(0)
            else:
                labels.append(0)

        if all(l == 0 for l in labels):
            labels[1] = tokens[1]
            tokens[1] = mask

        return torch.tensor(tokens), torch.tensor(labels)


class ImageTextMIMDataset(Dataset):
    def __init__(self,dataset,transform=None,patch_size=16,mask_ratio=0.75):
        self.dataset = dataset
        self.transform = transform
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]

        img = read_image(img_path).float()/255.0

        if self.transform is not None:
            img = self.transform(img) #! shape [3, H, W]

        patches = self._patchify(img) #! N, patch_dim

        masked_patches, mask, ids_restore = self._random_masking(patches, self.mask_ratio)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'original_image': img,
            'patches': patches,
            'masked_patches': masked_patches,
            'mask': mask,
            'ids_restore': ids_restore
        }

        return ret

    def _patchify(self, img):
        """
        Args:
            img: [3, H, W] tensor
        Returns:
            patches: [N, patch_dim]
        """
        C, H, W = img.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0
        h, w = H // self.patch_size, W // self.patch_size

        patches = img.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, C * self.patch_size * self.patch_size)
        return patches

    def _random_masking(self, patches, mask_ratio):
        """
        Args:
            patches: [N, D]
        Returns:
            x_masked: [N_visible, D]
            mask: [N], 0 for keep, 1 for mask
            ids_restore: [N]
        """
        N, D = patches.shape
        len_keep = int(N * (1 - mask_ratio))

        noise = torch.rand(N)
        ids_shuffle = torch.argsort(noise, dim=0)
        ids_restore = torch.argsort(ids_shuffle, dim=0)

        ids_keep = ids_shuffle[:len_keep]
        x_masked = patches[ids_keep]

        mask = torch.ones(N)
        mask[:len_keep] = 0
        mask = mask[ids_restore]

        return x_masked, mask, ids_restore
