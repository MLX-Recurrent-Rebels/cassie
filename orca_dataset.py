from torch.utils.data import Dataset
import transformers as t
import datasets as d
from transformers import AutoTokenizer

class OrcaDataset(Dataset):
    def __inicst__(self):
        self.tokeniser = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.tokeniser.pad_token_id = 0
        self.tokeniser.padding_side = "left"
        self.ds = d.load_dataset("Open-Orca/OpenOrca")
        self.ds = self.ds["train"]
        self.ds = self.ds.map(self.tokenise, remove_columns=["id"], load_from_cache_file=False, num_proc=8)
        
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

    def tokenise(self, elm):
        
        text = elm["system_prompt"] + elm["question"] + elm["response"]
        res = self.tokeniser(text)
        res["input_ids"].append(self.tokeniser.eos_token_id)
        res["attention_mask"].append(1)
        res["labels"] = res["input_ids"].copy()
        return res

    def max_seq_len(self):
        return max([len(elm["input_ids"]) for elm in self.ds])

