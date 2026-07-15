import torch
import torch.nn as nn

class JEPAEngine(nn.Module):
    def __init__(self, num_clases=2):
        super().__init__()
        # Codificador base (simplificado para alta velocidad)
        self.conv = nn.Conv2d(3, 16, kernel_size=16, stride=16)
        self.proj = nn.Linear(16, 768)
        
        # Saliencia (Periferia)
        self.compress = nn.Linear(768, 32)
        self.saliencia = nn.Linear(32, 1)
        
        # Fóvea y Memoria
        self.fovea = nn.Linear(768, 230)
        self.gru = nn.GRUCell(230, 128)
        self.head = nn.Linear(128, num_clases)

    def forward(self, x, num_sacadas=4):
        # 1. Periferia
        latente = torch.tanh(self.proj(self.conv(x).flatten(2).transpose(1, 2)))
        sal = torch.sigmoid(self.saliencia(torch.relu(self.compress(latente)))).squeeze(-1)
        _, idxs = torch.topk(sal, k=num_sacadas, dim=-1)
        
        # 2. Fóvea
        fovea_feat = torch.relu(self.fovea(x[:, idxs.squeeze(0), :]))
        h = torch.zeros(1, 128, device=x.device)
        for t in range(num_sacadas):
            h = self.gru(fovea_feat[:, t, :], h)
        return self.head(h), idxs
