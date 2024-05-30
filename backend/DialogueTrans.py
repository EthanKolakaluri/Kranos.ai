import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn
from torch.utils.data import DataLoader,TensorDataset
import torch.nn.utils.rnn as rnn
import torch.optim.lr_scheduler as lr_scheduler
from nltk.tokenize import word_tokenize
from torchtext.data import Field, TabularDataset

input_sequences = list()
output_sequences = list()
TEXT = Field(tokenize='spacy', init_token='<sos>',eos_token='<eos>', lower=True)

datafields = [('src', TEXT), ('trg', TEXT)]
dataset = TabularDataset(path='intents.json', format='json', fields=datafields)

TEXT.build_vocab(dataset, min_freq=2)

START_TOKEN = TEXT.vocab.stoi[TEXT.init_token]
END_TOKEN = TEXT.vocab.stoi[TEXT.eos_token]

class DialogueTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout):
        super(DialogueTransformer, self).__init__()
        self.encoder = TransformerEncoderLayer(input_dim, num_heads, dropout)
        self.decoder = TransformerDecoderLayer(output_dim, num_heads, dropout)
        self.fc = nn.Linear(input_dim, output_dim)
        self.positional_encoder = PositionalEncoding(input_dim)

    def forward(self, input_seq, output_seq=None):
        if output_seq is None:
            # Inference mode: generate output_seq
            input_seq = self.positional_encoder(input_seq)
            encoder_output = self.encoder(input_seq)
            output_seq = torch.full((input_seq.size(0), 1), START_TOKEN, device=input_seq.device)  # Start with the start-of-sequence token
            for _ in range(100):
                decoder_output = self.decoder(output_seq, encoder_output)
                next_token = self.fc(decoder_output[:, -1]).argmax(dim=-1).unsqueeze(-1)
                output_seq = torch.cat([output_seq, next_token], dim=-1)
                if next_token.item() == END_TOKEN:
                    break
            final_output = self.fc(decoder_output)
        else:
            # Training mode: use provided output_seq
            input_seq = self.positional_encoder(input_seq)
            encoder_output = self.encoder(input_seq)
            decoder_output = self.decoder(output_seq, encoder_output)
            final_output = self.fc(decoder_output)

        return final_output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq):
        attn_output, attn_weights = self.self_attn(input_seq, input_seq)
        attn_output = self.dropout(attn_output)
        ff_output = self.feed_forward(attn_output)
        return ff_output, attn_weights

class TransformerDecoderLayer(nn.Module):
    def __init__(self, output_dim, num_heads, dropout):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(output_dim, num_heads, dropout)
        self.encoder_attn = MultiHeadAttention(output_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, output_seq, encoder_output):
        attn_output, attn_weights = self.self_attn(output_seq, output_seq)
        attn_output = self.dropout(attn_output)
        encoder_attn_output, encoder_attn_weights = self.encoder_attn(attn_output, encoder_output)
        ff_output = self.feed_forward(encoder_attn_output)
        return ff_output

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key):
        queries = self.query_linear(query)
        keys = self.key_linear(key)
        values = self.value_linear(key)
        head_dim = queries.shape[-1] // 8
        queries = queries.view(queries.shape[0], queries.shape[1], 8, head_dim)
        keys = keys.view(keys.shape[0], keys.shape[1], 8, head_dim)
        values = values.view(values.shape[0], values.shape[1], 8, head_dim)
        attn_scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(head_dim)
        attn_scores = self.dropout(attn_scores)
        attn_weights = nn.Softmax(dim=-1)(attn_scores)
        output = torch.matmul(attn_weights, values)
        output = output.view(output.shape[0], output.shape[1], -1)
        return output, attn_weights

class PositionalEncoding(nn.Module):
    def __init__(self, input_dim):
        super(PositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.sin(torch.arange(input_dim) / (10000 ** (torch.arange(input_dim) / input_dim))))  # assuming max sequence length is 5000

def tokenize_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalnum()]  # Remove punctuation and convert to lowercase
    return tokens

sum = 0

with open('intents.json') as file:
    data = json.load(file)

for f in data["intents"]:
    input_sequences.append([len(f["patterns"])])
    output_sequences.append([len(f["responses"])])
    for i in f["responses"]:
       sum += len(tokenize_text(i))

print(sum)

# Assume you have a dataset of input and output sequences

padded_input = rnn.pad_sequence([torch.tensor(seq) for seq in input_sequences], batch_first=True, padding_value=0)
padded_output = rnn.pad_sequence([torch.tensor(seq) for seq in output_sequences],batch_first=True, padding_value = 0)

# Create a DataLoader
batch_size = 32

dataset = TensorDataset(padded_input, padded_output)
data_loader = DataLoader(dataset,batch_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
model = DialogueTransformer(input_dim=564, output_dim=1447, num_heads=8, dropout=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.075, weight_decay=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Train the model
for epoch in range(10):  # loop over the dataset multiple times
    for input_seq,output_seq in data_loader:
        input_seq, output_seq = input_seq.to(device), output_seq.to(device)
        optimizer.zero_grad()
        output = model(input_seq)
        loss = criterion(output, output_seq)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    # Validation
    model.eval()
    val_data_loader = DataLoader(TensorDataset(padded_input, padded_output), batch_size=batch_size, shuffle=True)
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in val_data_loader:
            input_seq, output_seq = batch
            input_seq, output_seq = input_seq.to(device), output_seq.to(device)
            output = model(input_seq)
            loss = criterion(output, output_seq)
            val_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == output_seq).sum().item()
    accuracy = correct / len(val_data_loader.dataset)
    print(f'Validation Loss: {val_loss / len(val_data_loader)}')
    print(f'Validation Accuracy: {accuracy:.4f}')