import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertModel, BertTokenizer
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
import pandas as pd

# Define the classifier
class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(768, num_classes)

    def forward(self, x):
        return self.linear(x)

# Define the BERT model
class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = Classifier(num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        logits = self.classifier(pooler_output)
        return logits

def encode_sentences(sentences, max_len = 128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

def train_bert_classifier(train_dataloader, val_dataloader, model, optimizer, num_epochs=25):


    final_loss = []
    final_val_loss = []

    print(f"Training Mode")
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch + 1))
        loss_list = []

        for step, batch in enumerate(train_dataloader):
            b_input_ids = batch[0]
            b_input_mask = batch[1]
            b_labels = batch[2].float()

            optimizer.zero_grad()
            outputs = model(b_input_ids, b_input_mask)
            loss = loss_fn(outputs, b_labels)
            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())

        avg_loss = torch.tensor(loss_list).mean().item()
        final_loss.append(avg_loss)
        print(f"Loss for Epoch: {avg_loss}")

        val_loss = test_bert_classifier(val_dataloader, model)
        final_val_loss.append(val_loss)
        print(f"Validation loss for epoch: {val_loss}")

    return final_loss, final_val_loss

def test_bert_classifier(dataloader, model):
    model.eval()
    test_loss = 0
    correct = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            b_input_ids = batch[0]
            b_input_mask = batch[1]
            b_labels = batch[2].float()

            outputs = model(b_input_ids, b_input_mask)
            loss = loss_fn(outputs, b_labels)

            test_loss = test_loss + loss
            val_preds = torch.argmax(outputs, dim=1)
            b_max = torch.argmax(b_labels, dim=1)
            correct = correct + torch.sum(b_max == val_preds).item()

    return 100 * correct / len(dataloader.dataset)

def predict(query, model):

    input_ids, attention_masks = encode_sentences([query], max_len=128)
    input_ids = input_ids
    attention_masks = attention_masks

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_masks)
        # print (outputs)
        outputs = outputs.reshape(outputs.shape[0], 9, 5)
        preds = torch.argmax(outputs, dim=-1)
        result_tensor = 5 - preds.numpy()
        # return result_tensor[0]
        print (result_tensor)
        return result_tensor.flatten()

def main():
    # Load data
    # df = pd.read_csv('temp.csv')
    # df = df.dropna(subset=['query'])
    # sentences = df['query'].values.tolist()
    # labels = df.iloc[:, 1:46].values
    # labels = labels.reshape(labels.shape[0], 9, 5)

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # input_ids, attention_masks = encode_sentences(sentences, max_len=128)

    # train_input_ids, val_input_ids, train_attention_mask, val_attention_mask, train_labels, val_labels = train_test_split(
    #     input_ids.numpy(), attention_masks.numpy(), labels, test_size=0.05
    # )

    # batch_size = 8
    # train_dataset = TensorDataset(
    #     torch.tensor(train_input_ids),
    #     torch.tensor(train_attention_mask),
    #     torch.tensor(train_labels)
    # )
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     sampler=RandomSampler(train_dataset),
    #     batch_size=batch_size
    # )

    # batch_size_val = 8
    # val_dataset = TensorDataset(
    #     torch.tensor(val_input_ids),
    #     torch.tensor(val_attention_mask),
    #     torch.tensor(val_labels)
    # )
    # val_dataloader = DataLoader(
    #     val_dataset,
    #     sampler=SequentialSampler(val_dataset),
    #     batch_size=batch_size_val
    # )

    # num_classes = 45
    # model = BertClassifier(num_classes=num_classes)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # # loss_fn = nn.CrossEntropyLoss(

    # final_loss, final_val_loss = train_bert_classifier(train_dataloader, val_dataloader, model, optimizer, num_epochs=25)

    # print("Training Losses:", final_loss)
    # print("Validation Losses:", final_val_loss)

    # # Save the trained model
    # torch.save(model.state_dict(), 'bert_classifier_model.pth')

    # # Load the trained model
    loaded_model = BertClassifier(num_classes=45)
    loaded_model.load_state_dict(torch.load('model2/best_model_checkpoint.pth'))

    feature_order = ['price', 'sim', 'processor', 'cam',  'ram', 'storage', 'battery', 'os', 'display']
    
    # Example queries for prediction
    queries = [
        "Looking for good cam phones",
        "Recommend phones with good battery life",
        "Suggest phones with both good camera and battery",
        "Smartphone with excellent camera performance",
        "Battery life comparison for latest phones",
        "Phones with top-notch camera technology",
        "Best battery backup phones",
        "Mobiles with superior camera quality",
        "Budget phone with ok battery life and average storage",
        "Phone with good ram and storage",
        "Need the best camera and top-notch battery life",
        "Affordable phone with no storage issues",
        "Looking for a phone with low as possible price and no problems",
        "Recommend best expensive phones",
        "Recommend best phones",
        "Phones with good display",
        "Best phone under 20K",
        "recommend phone with exceptional camera for capturing stunning photos",    
    ]

    feature_vectors = [predict(query.lower(), loaded_model) for query in queries]

    df = pd.DataFrame(feature_vectors, columns=feature_order)
    print(df)

if __name__ == "__main__":
    main()