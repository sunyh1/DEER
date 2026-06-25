import torch

f = '/path/to/checkpoint.pt'
ckpt = torch.load(f, map_location='cpu')
print(ckpt.keys())

state_dict = ckpt['state_dict']
print(state_dict.keys())
print(state_dict['module.visual.proj'].shape)
print(state_dict['module.visual.class_embedding'].shape)

state_dict['module.visual.proj'] = state_dict['module.visual.proj'][0]
state_dict['module.visual.class_embedding'] = state_dict['module.visual.class_embedding'][0]

torch.save(ckpt, f.replace('.pt', '_dr.pt'))
