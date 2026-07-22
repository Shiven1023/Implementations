
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self,seq_length, d,n_heads):
        super().__init__()
        self.seq_length = seq_length
        self.d = d
        self.n = n_heads
        self.in_proj = nn.Linear(d,3*d)            # (seq_len,3*d)
        self.out_proj = nn.Linear(d,d)
    def forward(self,x):
        x = self.in_proj(x)
        q,k,v = torch.split(x, split_size_or_sections=self.d, dim=1) # (seq_len,d ) , (seq_len,d) , (seq_len,d)
        q = q.reshape(self.seq_length,self.n,self.d//self.n)
        k = k.reshape(self.seq_length,self.n,self.d//self.n)
        v = v.reshape(self.seq_length,self.n,self.d//self.n) #n,seq_len , d' (using n as first dim as it is the batch_dim by deafult)
        q = q.permute(1,0,2)
        k = k.permute(1,0,2)
        v = v.permute(1,0,2)
        k_t = k.permute(0,2,1)
        attention  = torch.matmul(q,k_t)
        attention = attention/(self.d/self.n)**0.5
        mask = torch.triu(torch.ones(self.seq_length,self.seq_length),diagonal = 1)
        attention = attention.masked_fill(mask.bool(), float('-inf'))
        attention = torch.softmax(attention,dim = -1)
        score = attention @ v        #n,seq_len,seq_len . n,sewq_len,d' -->  n ,seq_len ,d'
        score = score.permute(1,0,2)
        score = score.reshape(self.seq_length, self.d)
        out = self.out_proj(score)
        return out

mha = MultiHeadAttention(seq_length=10, d=512, n_heads=8)
x = torch.randn(10, 512)
out = mha(x)
print(out.shape)


                                     
        


