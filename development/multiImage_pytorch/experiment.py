import matplotlib.pyplot as plt
import math
import shutil
import torch
from accelerate import Accelerator
from tensorboardX import SummaryWriter
from cli import parse_args
from dataset import SvbrdfDataset
from losses import MixedLoss, MixedLoss2
from models import MultiViewModel, SingleViewModel
from pathlib import Path
from persistence import Checkpoint
from renderers import LocalRenderer, RednerRenderer
import utils
import environment as env
import numpy as np


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

args = parse_args()

clean_training = args.mode == 'train' and args.retrain

# Load the checkpoint
checkpoint_dir = Path(args.model_dir)
checkpoint = Checkpoint()
if not clean_training:
    checkpoint = Checkpoint.load(checkpoint_dir)

# Immediatly restore the arguments if we have a valid checkpoint
if checkpoint.is_valid():
    args = checkpoint.restore_args(args)

# Make the result reproducible
utils.enable_deterministic_random_engine()

# Determine the device
accelerator = Accelerator()
device = accelerator.device

# Create the model
model = MultiViewModel(use_coords=args.use_coords).to(device)

if checkpoint.is_valid():
    model = checkpoint.restore_model_state(model)
elif args.mode == 'test':
    print("No model found in the model directory but it is required for testing.")
    exit(1)

# TODO: Choose a random number for the used input image count if we are training and we don't request it to be fix (see fixImageNb for reference)
data = SvbrdfDataset(data_directory=args.input_dir,
                     image_size=args.image_size, scale_mode=args.scale_mode, input_image_count=args.image_count, used_input_image_count=args.used_image_count,
                     use_augmentation=True, mix_materials=args.mode == 'train',
                     no_svbrdf=args.no_svbrdf_input, is_linear=args.linear_input)

epoch_start = 0

# model.generator.delete()


# model = torch.nn.Sequential(
#     *list(model.children())[:-8],
# )


# print(*list(model.parameters()))

if args.mode == 'train':
    validation_split = 0.01
    print("Using {:.2f} % of the data for validation".format(
        round(validation_split * 100.0, 2)))
    training_data, validation_data = torch.utils.data.random_split(data, [int(math.ceil(
        len(data) * (1.0 - validation_split))), int(math.floor(len(data) * validation_split))])
    print("Training samples: {:d}.".format(len(training_data)))
    print("Validation samples: {:d}.".format(len(validation_data)))

    training_dataloader = torch.utils.data.DataLoader(
        training_data, batch_size=8, pin_memory=True, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(
        validation_data, batch_size=8, pin_memory=True, shuffle=False)
    batch_count = int(math.ceil(len(training_data) /
                                training_dataloader.batch_size))

    # Train as many epochs as specified
    epoch_end = args.epochs

    print("Training from epoch {:d} to {:d}".format(epoch_start, epoch_end))

    # Set up the optimizer
    # TODO: Use betas=(0.5, 0.999)
    L = torch.FloatTensor(1, 3).uniform_(0.2, 1.0)
    
    L = L / torch.linalg.norm(L, ord=2, dim=-1, keepdim=True)
    
    L[:, :2] = 2.0 * L[:, :2] - 1.0

    V = torch.FloatTensor(1, 3).uniform_(0.2, 1.0)
    
    V = V / torch.linalg.norm(V, ord=2, dim=-1, keepdim=True)
    
    V[:, :2] = 2.0 * V[:, :2] - 1.0
    
    scene = env.generate_specific_scenes(1, L, V)
    L.requires_grad = True
    VIP = [L]
    
    # V.requires_grad = True
    optimizer = torch.optim.Adam(VIP, lr=0.1)
    model, optimizer, training_dataloader, validation_dataloader = accelerator.prepare(
        model, optimizer, training_dataloader, validation_dataloader)
    # print("scene", scene.camera)
    # TODO: Use scheduler if necessary
    #scheduler    = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

    # Set up the loss
    loss_renderer = LocalRenderer()

    loss_function = MixedLoss2(loss_renderer, scene[0])

    # Setup statistics stuff
    statistics_dir = checkpoint_dir / "logs"
    if clean_training and statistics_dir.exists():
        # Nuke the stats dir
        shutil.rmtree(statistics_dir)
    statistics_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(statistics_dir.absolute()))
    last_batch_inputs = None

    # Clear checkpoint in order to free up some memory
    checkpoint.purge()

    lights = []
    for epoch in range(epoch_start, epoch_end):
        for i, batch in enumerate(training_dataloader):
            # Unique index of this batch
            print("Ldet", (L.detach().numpy())[0])
            lights.append(((L.detach().numpy())[0]).tolist())
            scene = env.generate_specific_scenes(1, L, L)
            print("L", L)
            loss_function = MixedLoss2(loss_renderer, scene[0])
            batch_index = epoch * batch_count + i

            # Construct inputs
            batch_inputs = batch["inputs"].to(device)
            batch_svbrdf = batch["svbrdf"].to(device)

            # Perform a step
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = loss_function(outputs, batch_svbrdf)
            accelerator.backward(loss)
            optimizer.step()

            print("Epoch {:d}, Batch {:d}, loss: {:f}".format(
                epoch, i + 1, loss.item()))

            # Statistics
            writer.add_scalar("loss", loss.item(), batch_index)
            last_batch_inputs = batch_inputs
    lights.append(((L.detach().numpy())[0]).tolist())
    print("lights1", lights)
    print(len(lights))
    lights2 = []
    for j in range(len(lights)):
      if j%10 == 0:
        lights2.append(lights[j])
    # print("lights2", lights)
    # l=np.array(lights)
    l = np.array(lights2)
    print("lights3", l)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter([0.0], [0.0], [0.0], marker='o', c='r')


    # v = V.detach().numpy()
    ax.scatter(l[:,0], l[:,1], l[:,2], marker='.', c='g')
    # ax.scatter(v[:,0], v[:,1], v[:,2], marker='^', c='b')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(0., 3.)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.show()
    plt.savefig('/content/figures/experiment1.png')
    plt.show()
        # if epoch % args.save_frequency == 0:
        #     Checkpoint.save(checkpoint_dir, args, model, optimizer, epoch)

        # if epoch % args.validation_frequency == 0 and len(validation_data) > 0:
        #     model.eval()

        #     val_loss = 0.0
        #     batch_count_val = 0
        #     for batch in validation_dataloader:
        #         # Construct inputs
        #         batch_inputs = batch["inputs"].to(device)
        #         batch_svbrdf = batch["svbrdf"].to(device)

        #         outputs = model(batch_inputs)
        #         val_loss += loss_function(outputs, batch_svbrdf).item()
        #         batch_count_val += 1
        #     val_loss /= batch_count_val

        #     print("Epoch {:d}, validation loss: {:f}".format(epoch, val_loss))
        #     writer.add_scalar("val_loss", val_loss, epoch * batch_count)

            # model.train()

