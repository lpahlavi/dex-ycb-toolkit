import json

from dex_ycb_toolkit.factory import get_dataset


def main():
  for setup in ('s0', 's1', 's2', 's3'):
    for split in ('train', 'val', 'test'):
      name = '{}_{}'.format(setup, split)
      print('Dataset name: {}'.format(name))

      dataset = get_dataset(name)

      print('Dataset size: {}'.format(len(dataset)))

      sample = dataset[999]
      print('1000th sample:')
      print(json.dumps(sample, indent=4))


if __name__ == '__main__':
  main()
