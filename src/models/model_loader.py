from models.model import GraphNet

def load_model(config):
    model = GraphNet(input_dims=config.network.input_dims,
                    hidden_dim=config.network.hidden_dim,
                    output_dim=config.network.output_dim,
                    encoder_layers=config.network.encoder_layers,
                    decoder_layers=config.network.decoder_layers,
                    dropout_rate=config.network.dropout_rate,
                    knn_k=config.network.knn_k,
                    use_mha= config.network.use_mha,
                    num_mha_heads= config.network.num_mha_heads,
                    backbone=config.network.backbone,
                    mode=config.network.mode)

    return model

    
