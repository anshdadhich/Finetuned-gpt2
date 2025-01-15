import torch
from transformers import GPT2Tokenizer,GPT2LMHeadModel,DataCollatorWithPadding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = GPT2LMHeadModel.from_pretrained("gpt2.pt").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("GPT2tokenizer.pt")

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
