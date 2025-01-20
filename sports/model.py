import torch
import torch.nn as nn

class EventDetector(nn.Module):
    def __init__(self, hidden_dim, num_classes, num_layers=2, dropout=False):
        super(EventDetector, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Process spatial features at each time step using Conv1d
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),  # Flatten to [batch_size, features]
            nn.Linear(32 * 17, 128),  # Adjust as per output size of Conv1d
            nn.ReLU(),
        )
        
        # LSTM
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        
        if self.dropout:
            self.drop = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, 2, 17)
        
        batch_size, seq_length, channels, length = x.size()

        # Reshape x to [batch_size * seq_length, channels, length]
        x = x.view(-1, channels, length)
        
        # Process spatial features at each time step
        x = self.spatial_conv(x)
        # x shape: [batch_size * seq_length, 128]
        
        # Reshape x back to [batch_size, seq_length, 128]
        x = x.view(batch_size, seq_length, -1)
        
        if self.dropout:
            x = self.drop(x)
        
        # Pass through LSTM
        x, _ = self.lstm(x)
        
        if self.dropout:
            x = self.drop(x)
        
        # Apply fully connected layers to each time step
        x = self.fc_layers(x)
        # x shape: (batch_size, seq_length, num_classes)
        
        return x