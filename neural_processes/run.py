import torch
import numpy as np  # type: ignore
from db.db import DB  # type: ignore
from db.dataloader import CloseStockLoader  # type: ignore
from .np import NeuralProcess, context_target_split
from .trainer import NeuralProcessTrainer
from torch.utils.data import DataLoader
from matplotlib import animation, pyplot as plt  # type: ignore
from typing import Iterator, Tuple, NamedTuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fig, ax = plt.subplots(nrows=1, ncols=1)

GET_TOP_N = 1
SPAN = 1000  # 250 days of history
LABEL_SPAN = 20  # 7 days of labels
BATCH_SIZE = 32
X_DIM, Y_DIM = 1, 1
R_DIM, H_DIM, Z_DIM = 128, 128, 128
# CTX_RANGE = [100, SPAN]
# EXTRA_TARGET_RANGE = [0, LABEL_SPAN]
EPOCHS = 10
LR = 1e-4
MODEL_PATH = "./model.pt"

# the x context has to be the x position (in time) like 0 ~5000 something for the 20 years of data I have
# the y context has to be the prices

# the number of context points should be something random between 0 and 5000
# the target points should be the whole duration of time (including the future we want to predict)
# the loader should not have OHLCV prices, it should use close prices to start with and the complexity can move up from there


# The generator yield value for the frames function
class FramesReturn(NamedTuple):
    i: int
    x: torch.Tensor
    y: torch.Tensor
    x_front_half: torch.Tensor
    y_front_half: torch.Tensor


def train(db_path: str) -> None:
    """run a NP of a stock price prediction"""

    db = DB(db_path)

    train_dataset = CloseStockLoader(
        db, "1990-01-01", "2014-01-01", "AAPL", device=device, span=SPAN
    )

    print(f"length: {len(train_dataset)}")
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    # dimensions refers to the number of features in x and y, here x is only a number representing the x axis sequence
    # and y is 1 dimension (price)

    neural_process = NeuralProcess(X_DIM, Y_DIM, R_DIM, Z_DIM, H_DIM)
    neural_process.training = True

    optimizer = torch.optim.Adam(neural_process.parameters(), lr=LR)
    np_trainer = NeuralProcessTrainer(
        device, neural_process, optimizer, [100, SPAN], [100, SPAN], print_freq=100
    )

    np_trainer.train(train_loader, 20)
    torch.save(neural_process.state_dict(), MODEL_PATH)


def predict(db_path: str) -> None:
    db = DB(db_path)

    neural_process = NeuralProcess(X_DIM, Y_DIM, R_DIM, Z_DIM, H_DIM)
    neural_process.load_state_dict(torch.load(MODEL_PATH))

    test_dataset = CloseStockLoader(
        db, "2012-01-01", "2019-09-11", "AAPL", device=device, span=SPAN
    )
    test_loader = DataLoader(dataset=test_dataset)

    neural_process.training = False
    num_context, num_target = 50, 50

    def frames() -> Iterator[FramesReturn]:
        for i, batch in enumerate(test_loader):
            x, y = batch
            x_front_half, y_front_half = (
                x[:, : int(x.shape[1] / 2), :],
                y[:, : int(y.shape[1] / 2), :],
            )
            yield FramesReturn(i, x, y, x_front_half, y_front_half)

    # Extract a batch from data_loader
    def iterate(o: FramesReturn) -> None:
        # print(i)
        plt.cla()
        for i in range(64):
            # Use batch to create random set of context points
            x_context, y_context, x_target, y_target = context_target_split(
                o.x_front_half, o.y_front_half, num_context, num_target
            )

            # getting locations for the y's we are going to predict that are outside of the range of the context
            locations = np.random.choice(SPAN, size=num_target, replace=False)
            x_target = o.x[:, locations, :]
            y_target = o.y[:, locations, :]

            # Neural process returns distribution over y_target
            p_y_pred = neural_process(x_context, y_context, x_target)

            # Extract mean of distribution
            mu = p_y_pred.loc.detach()
            sigma = p_y_pred.scale.detach()

            plt.plot(x_target.numpy()[0], mu.numpy()[0], alpha=0.011, c="blue")
            plt.plot(
                x_target.numpy()[0],
                mu.numpy()[0] + sigma.numpy()[0],
                alpha=0.011,
                c="grey",
            )
            plt.plot(
                x_target.numpy()[0],
                mu.numpy()[0] - sigma.numpy()[0],
                alpha=0.01,
                c="grey",
            )
            plt.scatter(x_context[0].numpy(), y_context[0].numpy(), s=1, c="black")

    anim = animation.FuncAnimation(
        fig,
        iterate,
        frames=frames,
        interval=100,
        save_count=400,
        cache_frame_data=False,
    )
    anim.save("./anim.gif", writer="imagemagick")
