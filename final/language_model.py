from pycorrector import Corrector
import os
pwd_path = os.path.abspath(os.path.dirname(__file__))
lm_path = os.path.join(pwd_path, 'lm/zh_giga.no_cna_cmn.prune01244.klm')
model = Corrector(language_model_path=lm_path)

corrected_sent, detail = model.correct('少先队员因该为老人让坐')
print(corrected_sent, detail)
