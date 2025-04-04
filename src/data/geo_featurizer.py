import torch
import torch.nn as nn
import torch.nn.functional as F
from src.data.affine3d import build_affine3d_from_coordinates

def rbf_func(D, num_rbf):
    dtype = D.dtype
    device = D.device
    shape = D.shape
    D_min, D_max, D_count = 0., 20., num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count, dtype=dtype, device=device)
    D_mu = D_mu.view([1]*(len(shape))+[-1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF

def decouple(U, num_rbf):
    norm = U.norm(dim=-1, keepdim=True)
    mask = norm<1e-4
    direct = U/(norm+1e-6)
    direct[mask[...,0]] = 0
    rbf = rbf_func(norm[...,0], num_rbf)
    return torch.cat([direct, rbf], dim=-1)

class GeoFeaturizer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    @classmethod
    @torch.no_grad()
    def from_X_to_features(self, X):
        '''
        X: [N,L,4,3]
        '''
        T, mask = build_affine3d_from_coordinates(X[...,:3,:])
        T_ts = T[:,:,None].invert().compose(T[:,None])
        
        V, E = self.get_interact_feats(T, T_ts, X)
        
        return V, E, mask

    @classmethod
    @torch.no_grad()
    def get_interact_feats(self, T, T_ts, X):
        dtype = X.dtype
        device = X.device
        B, L, m, d = X.shape
        
        ## =========== node feature ===========
        diffX = F.pad(X.view(B,-1,d).diff(dim=1), (0,0,1,0)).view(B,L,m,d)
        diffX_proj = T[...,None].invert().rot.apply(diffX)
        V = decouple(diffX_proj, 16).view(B, L, -1)

        ## =========== pair feature ===========
        diffE = T[:,:,None,None].invert().apply(X[:,None,...])
        diffE = decouple(diffE, 16).view(B,L,L, -1)
        
        E_quant = T_ts.invert().rot._rots.reshape(B,L,L,9)
        E_trans = T_ts.trans
        E_trans = decouple(E_trans, 16).view(B,L,L,-1)
        
        E = torch.cat([diffE, E_quant, E_trans], dim=-1)
        return V.to(X.dtype), E.to(X.dtype)


if __name__ == '__main__':
    X = torch.rand(2, 10, 4, 3)
    geofeaturizer = GeoFeaturizer()
    V, E, attn_mask = geofeaturizer.from_X_to_features(X)
    print()

