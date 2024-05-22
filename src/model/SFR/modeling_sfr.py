import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from transformers import MistralForCausalLM,MistralModel


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class SFR(MistralModel):

    def get_embed_dim(self):
        return self.config.hidden_size
    
    def get_embed_length(self):
        return 1
    
    def get_embedding(self,input_ids,attention_mask):
        outputs = self.forward(input_ids=input_ids,attention_mask=attention_mask)
        embeddings = last_token_pool(outputs.last_hidden_state, attention_mask)
        return embeddings
    
    def get_doc_embedding(self,input_ids,attention_mask):
        return self.get_embedding(input_ids,attention_mask)
    
    def get_query_embedding(self,input_ids,attention_mask):
        return self.get_embedding(input_ids,attention_mask)


    # def get_detailed_instruct(task_description: str, query: str) -> str:
    #     return f'Instruct: {task_description}\nQuery: {query}'

    # # Each query must come with a one-sentence instruction that describes the task
    # task = 'Given a web search query, retrieve relevant passages that answer the query'
    # queries = [
    #     get_detailed_instruct(task, 'How to bake a chocolate cake'),
    #     get_detailed_instruct(task, 'Symptoms of the flu')
    # ]
    # # No need to add instruction for retrieval documents
    # passages = [
    #     "To bake a delicious chocolate cake, you'll need the following ingredients: all-purpose flour, sugar, cocoa powder, baking powder, baking soda, salt, eggs, milk, vegetable oil, and vanilla extract. Start by preheating your oven to 350°F (175°C). In a mixing bowl, combine the dry ingredients (flour, sugar, cocoa powder, baking powder, baking soda, and salt). In a separate bowl, whisk together the wet ingredients (eggs, milk, vegetable oil, and vanilla extract). Gradually add the wet mixture to the dry ingredients, stirring until well combined. Pour the batter into a greased cake pan and bake for 30-35 minutes. Let it cool before frosting with your favorite chocolate frosting. Enjoy your homemade chocolate cake!",
    #     "The flu, or influenza, is an illness caused by influenza viruses. Common symptoms of the flu include a high fever, chills, cough, sore throat, runny or stuffy nose, body aches, headache, fatigue, and sometimes nausea and vomiting. These symptoms can come on suddenly and are usually more severe than the common cold. It's important to get plenty of rest, stay hydrated, and consult a healthcare professional if you suspect you have the flu. In some cases, antiviral medications can help alleviate symptoms and reduce the duration of the illness."
    # ]

    # # load model and tokenizer
    # tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
    # model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral')

    # # get the embeddings
    # max_length = 4096
    # input_texts = queries + passages
    # batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
    # outputs = model(**batch_dict)
    # embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # # normalize embeddings
    # embeddings = F.normalize(embeddings, p=2, dim=1)
    # scores = (embeddings[:2] @ embeddings[2:].T) * 100
    # print(scores.tolist())
    # # [[86.7153549194336, 36.64569091796875], [35.00493621826172, 82.0738525390625]]
