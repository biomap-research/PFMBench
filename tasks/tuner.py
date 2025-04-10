import optuna
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

def objective(trial):
    # 可选不同的数据集或模型组合
    data_module = YourDataModule(dataset_name=trial.suggest_categorical("dataset", ["A", "B", "C"]))
    model = YourModel(learning_rate=trial.suggest_loguniform("lr", 1e-5, 1e-1))

    trainer = Trainer(
        max_epochs=10,
        logger=TensorBoardLogger("logs/", name="optuna"),
        callbacks=[],
    )

    trainer.fit(model, datamodule=data_module)

    return trainer.callback_metrics["val_loss"].item()  # 根据你实际的 validation metric 来写

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
