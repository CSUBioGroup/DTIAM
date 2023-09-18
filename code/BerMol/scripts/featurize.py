from bermol.trainer import BerMolPreTrainer


path = ""
predictor = BerMolPreTrainer.load(path)

def smi_feature(smi):
    output = predictor.transform([smi])
    pooled_output = output[0][1]
    return pooled_output.cpu().detach().numpy().reshape(-1)


smi = 'Cc1nc2c(c(-c3ccc(Cl)cc3Cl)c1CN)C(=O)N(CC(=O)N1CCCC1)C2'
vec = smi_feature(smi)
print(vec)
