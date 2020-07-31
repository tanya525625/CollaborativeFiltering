import os

from utils.models import MultiVAE
from utils.parser import parse_args
from utils.data_loader import DataLoader


def process_args(n_items):
    p_dims = eval(args.p_dims)
    q_dims = eval(args.q_dims)
    if q_dims:
        assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
        assert (
                q_dims[-1] == p_dims[0]
        ), "Latent dimension for p- and q- network mismatches."
    else:
        q_dims = p_dims[::-1]
    q_dims = [n_items] + q_dims
    p_dims = p_dims + [n_items]
    dropout_enc = eval(args.dropout_enc)
    dropout_dec = eval(args.dropout_dec)

    return p_dims, q_dims, dropout_enc, dropout_dec


if __name__ == "__main__":
    args = parse_args()
    DATA_DIR = "data"
    data_path = os.path.join(DATA_DIR,"_".join([args.dataset, "processed"]))
    model_name = str("_".join(["pt", args.model]))
    data_loader = DataLoader(data_path)
    n_items = data_loader.n_items

    p_dims, q_dims, dropout_enc, dropout_dec = process_args(n_items)
    model = MultiVAE(
        p_dims=p_dims,
        q_dims=q_dims,
        dropout_enc=dropout_enc,
        dropout_dec=dropout_dec,
    )
