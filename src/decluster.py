import pytorch_lightning as pl
from src.slow_eq import Catalog
import numpy as np
import eq
import torch
import warnings


def time_torchETAS_decluster(
    event_catalog: Catalog,
    burn_in: float = 0.1,
) -> Catalog:
    t_start = 0
    t_end = event_catalog.duration / np.timedelta64(1, "D")

    inter_times = np.diff(
        (
            (event_catalog.catalog.time - event_catalog.start_time)
            / np.timedelta64(1, "D")
        ).values,
        prepend=t_start,
        append=t_end,
    )

    seq = eq.data.Sequence(
        inter_times=torch.as_tensor(inter_times, dtype=torch.float32),
        t_start=t_start,
        t_nll_start=t_end * burn_in,
        t_end=t_end,
        mag=torch.as_tensor(event_catalog.catalog.mag.values, dtype=torch.float32),
    )
    dataset = eq.data.InMemoryDataset(sequences=[seq])
    dl = dataset.get_dataloader()

    model = eq.models.ETAS(
        mag_completeness=event_catalog.mag_completeness,
        base_rate_init=len(event_catalog) / t_end,
    )

    trainer = pl.Trainer(
        max_epochs=400,
        devices=1,
        accelerator="mps",
        enable_checkpointing=False,
        logger=False,
    )

    # turn off pytorch lightning warning messages:
    # Filter annoying warnings by PytorchLightning
    warnings.filterwarnings(
        "ignore", ".*You defined a `validation_step` but have no `val_dataloader`*"
    )
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    warnings.filterwarnings("ignore", ".*has `shuffle=True`, it is strongly*")

    trainer.fit(model, dl)

    parameter_str = ["mu", "k", "c", "p", "alpha"]

    mu, k, c, p, alpha = [getattr(model, param).item() for param in parameter_str]

    [print(f"{param}: {getattr(model, param).item(): .2f}") for param in parameter_str]

    etas_rate = lambda t, ti, mi: mu + np.sum(
        k * 10 ** (alpha * (mi - event_catalog.mag_completeness)) / (t - ti + c) ** p
    )
    rate = np.array(
        [
            etas_rate(
                t,
                seq.arrival_times[seq.arrival_times < t].detach().numpy(),
                seq.mag[seq.arrival_times < t].detach().numpy(),
            )
            for t in seq.arrival_times.detach().numpy()
        ]
    )

    background_probability = model.mu.item() / rate

    thinned_bool = [
        True if np.random.rand() < p else False for p in background_probability
    ]
    thinned_event_catalog = event_catalog[thinned_bool]

    return Catalog(thinned_event_catalog)
