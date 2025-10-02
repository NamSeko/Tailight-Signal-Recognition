import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder # type: ignore
from torchvision import transforms

from crnn.functions import DecoderRNN_varlen, ResCNNEncoder, load_config


class ResNetLSTMPredictor:
    def __init__(self, config_path="./config.yaml"):
        self.config = load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_names = list(self.config["action_names"])
        self.res_size = 224

        self.le = LabelEncoder()
        self.le.fit(self.action_names)

        self.transform = transforms.Compose([
            transforms.Resize([self.res_size, self.res_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self._load_model()

    def _load_model(self):
        k = len(self.action_names)
        cnn_h1, cnn_h2, cnn_embed = 768, 512, 256
        rnn_layers, rnn_nodes, rnn_fc = 3, 256, 128
        drop_p = 0.2

        self.cnn_encoder = ResCNNEncoder(
            fc_hidden1=cnn_h1,
            fc_hidden2=cnn_h2,
            drop_p=drop_p,
            CNN_embed_dim=cnn_embed
        ).to(self.device)

        self.rnn_decoder = DecoderRNN_varlen(
            CNN_embed_dim=cnn_embed,
            h_RNN_layers=rnn_layers,
            h_RNN=rnn_nodes,
            h_FC_dim=rnn_fc,
            drop_p=drop_p,
            num_classes=k
        ).to(self.device)

        self.cnn_encoder.load_state_dict(torch.load(self.config["cnn_encoder"], map_location=self.device))
        self.rnn_decoder.load_state_dict(torch.load(self.config["rnn_decoder"], map_location=self.device))

        self.cnn_encoder.eval()
        self.rnn_decoder.eval()

    def predict(self, frames_list):
        with torch.no_grad():
            num_frames = len(frames_list)
            X = torch.zeros((num_frames, 3, self.res_size, self.res_size))

            for i, frame in enumerate(frames_list):
                frame_tensor = self.transform(frame)
                X[i] = frame_tensor

            X = X.unsqueeze(0).to(self.device)
            video_len_tensor = torch.LongTensor([num_frames]).to(self.device)

            features = self.cnn_encoder(X)
            output = self.rnn_decoder(features, video_len_tensor)

            prob = F.softmax(output, dim=1)
            conf, pred_class = torch.max(prob, dim=1)
            pred_label = self.le.inverse_transform([pred_class.item()])[0]
            confidence = conf.item()
            return pred_label, confidence

if __name__ == "__main__":
    import glob
    import os

    from PIL import Image
    
    folder_path = "dataset/left_signal/demo_motor_1146_round_1"
    image_paths = sorted(glob.glob(folder_path + "/*.jpg"), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frames = [Image.open(p) for p in image_paths]

    predictor = ResNetLSTMPredictor(config_path="./config.yaml")
    label, conf = predictor.predict(frames)
    print('Folder path:', folder_path)
    print('Frame length:', len(frames))
    print("Predicted action:", label, round(conf, 2), '\n')
