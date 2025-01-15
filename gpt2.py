import torch
from transformers import GPT2Tokenizer,GPT2LMHeadModel,DataCollatorWithPadding
from torch.utils.data import Dataset,DataLoader
import pandas
import matplotlib.pyplot as mat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = GPT2LMHeadModel.from_pretrained("gpt2.pt").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("GPT2tokenizer.pt")

dataset = pandas.read_csv(r"Conversation.csv",usecols = ["question","answer"])

questions = dataset["question"]
answers = dataset["answer"]

inputt = questions + " <SEP> " + answers

inputt = inputt.to_numpy()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2") 
data_collator = DataCollatorWithPadding(tokenizer)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token" : "<PAD>"})
    
if "<SEP>" not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({"additional_special_tokens" : ["<SEP>"]})
 
class custom_dataset(Dataset):
      def __init__(self,inputt,tokenizer):
          super().__init__() 
          self.inputt = inputt
          self.tokenizer = tokenizer

      def __len__(self):
          return(len(self.inputt))
      
      def __getitem__(self,index):
          return self.inputt[index] 

def collate_function(batch,tokenizer):
    inputs = batch
    inputs = tokenizer(inputs, padding = True, truncation = True, return_tensors = "pt")
    return {key : value.to(device) for key,value in inputs.items()}
             
dataset = custom_dataset(inputt,tokenizer)

training_data = DataLoader(dataset,batch_size = 32, shuffle = True, collate_fn = lambda x : collate_function(x,dataset.tokenizer))

model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.resize_token_embeddings(len(tokenizer))

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)

losses = []

for epoch in range(20):
    for batch,(inputt) in enumerate(training_data):
        model.train()
        optimizer.zero_grad()
        
        output = model(**inputt, labels = inputt["input_ids"])
        loss = output.loss

        masked_loss = loss * inputt["attention_mask"]

        valid_tokens = inputt["attention_mask"].sum()
        loss = masked_loss.sum()/valid_tokens
                
        loss.backward()
        optimizer.step()
        
    logits = output.logits
    predicted = torch.argmax(logits,dim = -1)
    print(tokenizer.decode(predicted[0]))
     
    losses.append(loss.item())
    print(f"loss at epoch {epoch} is {loss.item()}")
    
mat.plot(range(len(losses)),losses)
mat.xlabel = "epochs"
mat.ylabel = "loss"
mat.savefig("loss.jpg")

model.save_pretrained("gpt2.pt")
tokenizer.save_pretrained("GPT2tokenizer.pt")

with torch.no_grad():
     model.eval()
     question = "hi how are you doing <SEP>"
     question = tokenizer(question,return_tensors = "pt") 
     question = {key : value.to(device) for key,value in question.items()}
     question_ids = question["input_ids"]
     
     for i in range(10):
         output = model(question_ids)
         predicted_word_logits = output.logits[:,-1,:]
         predicted_word_id = torch.argmax(predicted_word_logits,dim = -1)
                 
         if predicted_word_id.item() == tokenizer.eos_token_id:
             break
         
         question_ids = torch.cat([question_ids,predicted_word_id.unsqueeze(0)],dim = -1)
         
     print("answer : ", tokenizer.decode(question_ids[0]))

