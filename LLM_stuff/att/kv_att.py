
import torch
import torch.nn as nn
"""
if use_cache = true --> input x is single token (1,d) so q = (1,d) k = (seq_len , d) (after torch.cat) v = (seq_len,d)
q --> (1,n,d')-->(n, 1 , d') , v -- (n ,seq_len , d') k -->(n, seq_len , d')
q.kT = (n, 1 , seq_len) (scores for last token wrt to each token)(no use for mask)
softmax(q.kT/ROOT(D)) . V --> (n, 1, d') logit for last token --> (1,d) return 

if use_cache is true then x is full sequence
q.kT --> (n , seq_len , seq_len) (mask needed upper triangle))
softmax(q.kT/root(d)) .V --> n , seq_len , d' --> seq_len, d --> (logits for full sequence)


"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d,n_heads ):
        super().__init__()
        self.d = d
        self.register_buffer("k_cache", None)
        self.register_buffer("v_cache", None)
        self.n = n_heads
        self.Q = nn.Linear(d,d)      
        self.K = nn.Linear(d,d)      
        self.V = nn.Linear(d,d)
        self.out_proj = nn.Linear(d,d)
    def forward(self,x, use_cache = False):
        seq_len = x.shape[0]
        q = self.Q(x)
        new_k = self.K(x)
        new_v = self.V(x)
        if use_cache == True:
            if self.k_cache is None:
                self.k_cache , self.v_cache = new_k , new_v
            else:
                self.k_cache = torch.cat([self.k_cache , new_k], dim = 0) # (seq_len , d)
                self.v_cache = torch.cat([self.v_cache , new_v], dim = 0)
            k , v = self.k_cache , self.v_cache
        else:
            k , v = new_k , new_v
        q = q.view(seq_len,self.n,self.d//self.n).permute(1,0,2)
        k = k.view(k.shape[0],self.n,self.d//self.n).permute(1,0,2)
        v = v.view(v.shape[0],self.n,self.d//self.n).permute(1,0,2)
        score = q @(k.transpose(-1,-2))
        score = score/ ((self.d/self.n)**0.5)
        if use_cache == True: #(score --> (n,1,d'))
            score = torch.softmax(score,dim = -1)
            attention = score @ v
            attention = attention.permute(1,0,2)
            attention = attention.reshape(1,self.d)
            out = self.out_proj(attention)
            return out
        else:
            mask = torch.triu(torch.ones(seq_len,seq_len),diagonal = 1)
            score = score.masked_fill(mask.bool(), float('-inf'))
            score = torch.softmax(score,dim = -1)
            attention = score @ v        #n,seq_len,seq_len . n,sewq_len,d' -->  n ,seq_len ,d'
            attention = attention.permute(1,0,2)
            attention = attention.reshape(seq_len, self.d)
            out = self.out_proj(attention)
            return out


            




        
        """q,k,v = torch.split(x, split_size_or_sections=self.d, dim=1) # (seq_len,d ) , (seq_len,d) , (seq_len,d)
        q = q.view(self.seq_length,self.n,self.d//self.n).permute(1,0,2)
        k = k.view(self.seq_length,self.n,self.d//self.n).permute(1,0,2)
        v = v.view(self.seq_length,self.n,self.d//self.n).permute(1,0,2) #n,seq_len , d' (using n as first dim as it is the batch_dim by deafult)
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
        return out"""

                                     
        


