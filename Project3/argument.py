def add_arguments(parser):
    """
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument(
        "--total_frames", type=int, default=5_000_000, help="total number of frames"
    )

    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size for training"
    )
    parser.add_argument("--train_freq", type=int, default=4, help="training frequency")
    parser.add_argument(
        "--replay_buffer_size", type=int, default=1000000, help="replay buffer size"
    )
    parser.add_argument(
        "--replay_start_size",
        type=int,
        default=50000,
        help="number of frames to collect before updating policy",
    )

    parser.add_argument(
        "--target_update_freq",
        type=int,
        default=10000,
        help="target network update frequency",
    )

    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")

    parser.add_argument(
        "--eps_start",
        type=float,
        default=1.0,
        help="initial epsilon for epsilon-greedy",
    )
    parser.add_argument(
        "--eps_end", type=float, default=0.1, help="final epsilon for epsilon-greedy"
    )
    parser.add_argument(
        "--eps_decay", type=int, default=1000000, help="epsilon decay steps"
    )

    return parser
