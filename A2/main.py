from mnist import MNIST
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--run_mode', type=str, default='train', choices=['train', 'flip_test', 'gaussian_test'])
    parser.add_argument('--augment', action='store_true')
    

    args = parser.parse_args()
    print(args)
    mnist = MNIST(epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate, augment=args.augment)
    if args.run_mode == 'train':
        mnist.train()
        mnist.plot_metrics()
    elif args.run_mode == 'flip_test':
        mnist.test_with_flips()
        mnist.plot_flip()
    elif args.run_mode == 'gaussian_test':
        mnist.test_with_gaussian_noise()
        mnist.plot_gaussian_noise()
    else:
        raise ValueError('Invalid run mode')
    