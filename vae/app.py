from typing import TYPE_CHECKING, Any

import os

import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import dash
from dash import dcc, html
from dash import Input, Output, State
import dash_bootstrap_components as dbc

from vae.model import VAE

# from datasets import MNISTDataset
# from resolve_dataset import mnist
from vae.train import train
from vae.configs import ModelConfig, TrainConfig


pio.templates.default = "plotly_dark"


class App:

    def __init__(self) -> None:

        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
        )

        self.app.layout = self.layout()

        # Datasets
        # dataset = mnist("data")
        # self.train_dataset = MNISTDataset(
        #     imgs=dataset["train_images"], labels=dataset["train_labels"]
        # )
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert image to tensor
                transforms.Lambda(
                    lambda x: x.view(-1)
                ),  # Flatten the image (28x28 -> 784)
            ]
        )
        self.train_dataset = datasets.MNIST(
            "data", train=True, download=True, transform=transform
        )

        self.model: VAE | None = None

        self.register_callbacks()

    def run(self, port: str = "8050", debug: bool = False) -> None:
        self.app.run(port=port, debug=debug)

    def create_new_model(self, config: dict[str, Any]) -> None:
        # Calculate input size
        sample, _ = self.train_dataset[0]
        n_in = len(sample)

        model = VAE(
            n_in=n_in,
            n_latent=config["latent_size"],
            hidden_dims=config["hidden_layers"],
        )

        self.model = model

    def register_callbacks(self) -> None:

        @self.app.callback(
            Output("configs-modal", "is_open"),
            Input("configs-button", "n_clicks"),
        )
        def open_config_modal(n_clicks: int) -> bool:
            """Open config modal."""
            if n_clicks:
                return True
            return False

        @self.app.callback(
            Output("model-config-store", "data"),
            Input("latent-size-input", "value"),
            Input("hidden-layers-input", "value"),
        )
        def update_model_config(latent_size: int, hidden_layers: str) -> dict[str, Any]:
            # TODO: Input validation

            # Process hidden layers to tuple of ints
            layers = tuple(int(n.strip()) for n in hidden_layers.split(","))

            return {
                "latent_size": latent_size,
                "hidden_layers": layers,
            }

        @self.app.callback(
            Output("train-config-store", "data"),
            Input("learning-rate-input", "value"),
            Input("batch-size-input", "value"),
            Input("steps-input", "value"),
            Input("logging-steps-input", "value"),
        )
        def update_train_config(
            lr: float,
            batch_size: int,
            steps: int,
            logging_steps: int,
        ) -> dict[str, Any]:
            # TODO: Input validation

            return {
                "lr": lr,
                "batch_size": batch_size,
                "steps": steps,
                "logging_steps": logging_steps,
            }

        @self.app.callback(
            Output("train-metrics-store", "data"),
            Input("start-train-button", "n_clicks"),
            State("train-config-store", "data"),
            State("model-config-store", "data"),
            prevent_initial_call=True,
        )
        def train_model(
            n_clicks: int,
            _train_config: dict[str, Any],
            _model_config: dict[str, Any],
        ) -> dict:
            # Create model
            self.create_new_model(_model_config)
            assert self.model is not None

            # Wrap train config
            train_config = TrainConfig(
                lr=_train_config["lr"],
                batch_size=_train_config["batch_size"],
                steps=_train_config["steps"],
                logging_steps=_train_config["logging_steps"],
            )

            train_stats = train(
                model=self.model,
                train_dataset=self.train_dataset,
                test_dataset=self.train_dataset,
                train_config=train_config,
            )

            return train_stats

        @self.app.callback(
            Output("loss-fig", "figure"),
            Input("train-metrics-store", "data"),
            prevent_initial_call=True,
        )
        def plot_losses(metrics: dict[str, Any] | None) -> go.Figure:
            if metrics is None:
                return go.Figure()
            steps = metrics["steps"]
            train_loss = metrics["train_loss"]
            train_recon_loss = metrics["train_recon_loss"]
            train_kl_loss = metrics["train_kld_loss"]

            fig = make_subplots(
                rows=1,
                cols=1,
                shared_xaxes=True,
            )

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=train_loss,
                    mode="lines",
                    name="Total loss",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=train_recon_loss,
                    mode="lines",
                    name="Reconstruction loss",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=train_kl_loss,
                    mode="lines",
                    name="KL loss",
                ),
                row=1,
                col=1,
            )

            fig.update_layout(
                title="Losses",
                title_x=0.5,
                xaxis_title="Steps",
                yaxis_title="Loss",
            )

            return fig

        @self.app.callback(
            Output("samples-fig", "figure"),
            Input("train-metrics-store", "data"),
        )
        def update_samples_fig(metrics: dict[str, Any] | None) -> go.Figure:
            if metrics is None or self.model is None:
                return go.Figure()

            # Select random samples from training data
            n = 4
            # idxs = np.random.choice(len(self.train_dataset), n)
            # samples = self.train_dataset[idxs][0]
            dataloader = DataLoader(self.train_dataset, batch_size=n, shuffle=True)
            samples = next(iter(dataloader))[0]
            reconstructions = self.model.generate(samples)

            # Convert data to numpy arrays and reshape
            samples = samples.detach().numpy().reshape(n, 28, 28) * 255
            reconstructions = reconstructions.detach().numpy().reshape(n, 28, 28) * 255

            fig = make_subplots(
                rows=2,
                cols=n,
                horizontal_spacing=0.1,
                vertical_spacing=0.1,
                row_titles=["Original", "Reconstruction"],
            )

            for i in range(n):
                im1 = samples[i].reshape(28, 28, 1).repeat(3, axis=-1)
                im2 = reconstructions[i].reshape(28, 28, 1).repeat(3, axis=-1)
                fig.add_trace(
                    go.Image(z=im1),
                    row=1,
                    col=i + 1,
                )
                fig.add_trace(
                    go.Image(z=im2),
                    row=2,
                    col=i + 1,
                )

            fig.update_layout(
                title="Samples",
                title_x=0.5,
            )

            return fig

    def layout(self) -> html.Div:
        layout = html.Div(
            children=[
                dbc.Tabs(
                    children=[
                        dbc.Tab(
                            children=self._training_tab(),
                            label="Tab One",
                        ),
                        dbc.Tab(label="Tab Two"),
                    ],
                ),
            ],
        )
        return layout

    def _training_tab(self) -> html.Div:
        layout = html.Div(
            children=[
                dcc.Store(
                    id="train-metrics-store",
                    data=None,
                ),
                dcc.Store(
                    id="model-config-store",
                    data=None,
                ),
                dcc.Store(
                    id="train-config-store",
                    data=None,
                ),
                html.Div(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(
                                id="loss-fig",
                                figure=go.Figure(),
                            ),
                        ),
                    ),
                    style={
                        "grid-area": "loss-graph",
                    },
                ),
                html.Div(
                    dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(
                                id="samples-fig",
                                figure=go.Figure(),
                            ),
                        ),
                    ),
                    style={
                        "grid-area": "samples-graph",
                    },
                ),
                html.Div(
                    [
                        dbc.Button(
                            "Configs",
                            id="configs-button",
                        ),
                        dbc.Modal(
                            [
                                dbc.ModalHeader(dbc.ModalTitle("Configs")),
                                dbc.ModalBody(self._configs_div()),
                            ],
                            id="configs-modal",
                            is_open=False,
                        ),
                        dbc.Button(
                            "Train",
                            id="start-train-button",
                        ),
                    ],
                    style={
                        "grid-area": "buttons",
                        # "display": "flex",
                    },
                    className="gap-10",
                ),
            ],
            style={
                "display": "grid",
                "grid-template-columns": "4fr 1fr",
                "grid-template-rows": "1fr 1fr",
                "grid-gap": "20px",
                "padding": "20px",
                "grid-template-areas": """
                    "loss-graph buttons"
                    "samples-graph buttons"
                """,
            },
        )

        return layout

    def _configs_div(self) -> html.Div:
        layout = html.Div(
            children=[
                dbc.Tabs(
                    [
                        dbc.Tab(
                            [
                                html.Div(
                                    id="model-configs-div",
                                    children=[
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Latent size"),
                                                dbc.Input(
                                                    id="latent-size-input",
                                                    value=2,
                                                    type="number",
                                                    min=1,
                                                    step=1,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Hidden Layers"),
                                                dbc.Input(
                                                    id="hidden-layers-input",
                                                    value="100, 50",
                                                    type="str",
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                    ],
                                )
                            ],
                            label="Model",
                        ),
                        dbc.Tab(
                            [
                                html.Div(
                                    id="training-configs-div",
                                    children=[
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Learning rate"),
                                                dbc.Input(
                                                    id="learning-rate-input",
                                                    value=0.0001,
                                                    min=0.0,
                                                    type="number",
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Batch size"),
                                                dbc.Input(
                                                    id="batch-size-input",
                                                    value=32,
                                                    min=1,
                                                    step=1,
                                                    type="number",
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Steps"),
                                                dbc.Input(
                                                    id="steps-input",
                                                    value=10000,
                                                    min=1,
                                                    step=1,
                                                    type="number",
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        dbc.InputGroup(
                                            [
                                                dbc.InputGroupText("Logging steps"),
                                                dbc.Input(
                                                    id="logging-steps-input",
                                                    value=10,
                                                    min=1,
                                                    step=1,
                                                    type="number",
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                    ],
                                )
                            ],
                            label="Training",
                        ),
                    ]
                )
            ],
        )
        return layout


if __name__ == "__main__":
    app = App()
    app.run(debug=True)
