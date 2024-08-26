# -*- coding:utf-8 -*-
# @FileName  :test.py
# @Time      :2024/6/1 下午6:10
# @Author    :ZMFY
# Description:


from config import Config
from dataloader.loader import init_dataloader
from networks.mmfd import MultiModalFD, UniModalFD
from trainer import MMFDTrainer
from utils import *

base_dir = "/opt/data/private/Meta-Learning/FaultDiagnosis/MMPN-FD"
data_name = 'CWRU'
assert os.path.exists(base_dir)
timestamp = "08-25-23-29"
read_best = False  # 读取best模型还是final模型
log_name = 'test'
trained_model_dir = os.path.join(base_dir, 'output', data_name, timestamp)
set_logger(trained_model_dir, log_name)

if read_best:
    trained_model_path = os.path.join(trained_model_dir, 'model_best.pt')
else:
    trained_model_path = os.path.join(trained_model_dir, 'model_final.pt')
config_path = os.path.join(trained_model_dir, 'config.json')
assert os.path.isfile(config_path)

logging.info("loading config and dataloader...")
config = Config.init_from_json(config_path)
set_seed(config.seed)
print(vars(config))
dataloader = init_dataloader(config, mode='test')

set_seed(config.seed)

logging.info(f"loading trained model from {trained_model_path}")

modal = config.modal
input_shape = config.input_shape

device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
if modal == 'all':
    model = MultiModalFD(config, input_shape, device=device)
else:
    model = UniModalFD(config, input_shape, modal=modal)
state = torch.load(trained_model_path)
model.load_state_dict(state)

logging.info("Run testing...")
loss, acc = MMFDTrainer.evaluate(model, dataloader, config.n_way, config.k_shot_eval,
                                 config.n_query_eval, device, desc='Test', modal=config.modal)
logging.info(f"\nTest Done")
logging.info(f"Dataset: {data_name}")
logging.info(f"task name: {config.task_name}")
logging.info(f"{config.n_way}-Way-{config.k_shot_eval}-Shot")
logging.info(f"Loss: {loss:.4f}, Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    pass
