import warnings
import torch 
from utils.torch_utils import reduce_across_processes
from utils.metrics import ConfusionMatrix, MetricLogger

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"
    num_processed_samples = 0
    with torch.inference_mode():
        for batch in metric_logger.log_every(data_loader, 100, header):
            if len(batch) == 3:
                image, target, fname = batch
            else:
                image, target = batch
                fname = None
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output["out"]

            confmat.update(target.flatten(), output.argmax(1).flatten())
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            num_processed_samples += image.shape[0]

        confmat.reduce_from_all_processes()

    num_processed_samples = reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    return confmat
