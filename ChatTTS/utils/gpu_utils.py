import torch
import logging


def select_device(min_memory=2048):
    logger = logging.getLogger(__name__)
    if torch.cuda.is_available():
        available_gpus = []
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.zeros(1).to(
                torch.device(f"cuda:{i}")
            )  # 分配一个小张量到GPU以强制初始化
            props = torch.cuda.get_device_properties(i)
            free_memory = (
                props.total_memory
                - torch.cuda.memory_reserved(i)
                - torch.cuda.memory_allocated(i)
            )
            available_gpus.append((i, free_memory))
            logger.log(
                logging.WARNING,
                f"GPU {i}/{torch.cuda.device_count()} has {round(free_memory, 2)} MB memory left. reserved: {torch.cuda.memory_reserved(i)}, allocated: {torch.cuda.memory_allocated(i)}",
            )
        selected_gpu, max_free_memory = max(available_gpus, key=lambda x: x[1])
        device = torch.device(f"cuda:{selected_gpu}")
        free_memory_mb = max_free_memory / (1024 * 1024)
        if free_memory_mb < min_memory:
            logger.log(
                logging.WARNING,
                f"GPU {selected_gpu} has {round(free_memory_mb, 2)} MB memory left.",
            )
            device = torch.device("cpu")
    else:
        logger.log(logging.WARNING, f"No GPU found, use CPU instead")
        device = torch.device("cpu")

    return device
