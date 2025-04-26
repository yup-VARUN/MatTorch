# Parser for my Domain Specific Language: "MatTorch"

def parse_dsl(dsl_str):
    """
    Parses a MatTorch DSL string into a specification dict:
      - name: network name
      - input_dim: integer
      - layers: list of {'output_dim', 'activation'}
      - dataset: {'name', 'batch_size', 'shuffle'}
      - validation: float ratio
      - optimizer: string
      - loss: string
      - train: {'epochs', 'batch_size'}
    """
    spec = {
        'layers': []
    }
    for line in dsl_str.strip().splitlines():
        tokens = line.strip().split()
        if not tokens:
            continue
        key = tokens[0].lower()
        if key == 'network':
            spec['name'] = tokens[1]
        elif key == 'input':
            spec['input_dim'] = int(tokens[1])
        elif key == 'layer':
            spec['layers'].append({
                'output_dim': int(tokens[1]),
                'activation': tokens[2].lower()
            })
        elif key == 'dataset':
            # e.g. dataset mnist batch_size 64 shuffle true
            spec['dataset'] = {
                'name': tokens[1].lower(),
                'batch_size': int(tokens[tokens.index('batch_size') + 1]),
                'shuffle': tokens[tokens.index('shuffle') + 1].lower() == 'true'
            }
        elif key == 'validation':
            spec['validation'] = float(tokens[1])
        elif key == 'optimizer':
            spec['optimizer'] = tokens[1].lower()
        elif key == 'loss':
            spec['loss'] = tokens[1].lower()
        elif key == 'train':
            # e.g. train epochs 10 batch_size 32
            spec['train'] = {
                'epochs': int(tokens[tokens.index('epochs') + 1]),
                'batch_size': int(tokens[tokens.index('batch_size') + 1])
            }
        else:
            raise ValueError(f"Unknown DSL keyword: {tokens[0]}")
    return spec