import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json

from models import HRM_4x4 
from generators import generate_dataset, print_puzzle


class Dataset_4x4(Dataset):
    def __init__(self, puzzles: np.ndarray, solutions: np.ndarray):
        self.puzzles = torch.from_numpy(puzzles).long()
        
        # Target: first empty cell and its digit
        self.targets = []
        for puzzle, solution in zip(puzzles, solutions):
            empty = np.argwhere(puzzle == 0)
            if len(empty) > 0:
                r, c = empty[0]
                cell_idx = r * 4 + c
                digit = solution[r, c] - 1  # 0-indexed
            else:
                cell_idx = 0
                digit = 0
            self.targets.append((cell_idx, digit))
    
    def __len__(self):
        return len(self.puzzles)
    
    def __getitem__(self, idx):
        return self.puzzles[idx], self.targets[idx][0], self.targets[idx][1]


def train_model(num_epochs: int = 20, 
                train_size: int = 1000,
                hidden_dim: int = 64,
                max_iterations: int = 10,
                halt_weight: float = 0.1):
    """
    Train the HRM 4x4 solver
    
    Args:
        num_epochs: Number of training epochs
        train_size: Number of training puzzles
        hidden_dim: Hidden dimension size
        max_iterations: Maximum Worker iterations
        halt_weight: Weight for halting penalty
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}\n")
    
    # Generate data
    print("Generating training data...")
    train_puzzles, train_solutions = generate_dataset(train_size)
    val_puzzles, val_solutions = generate_dataset(100)
    
    train_data = Dataset_4x4(train_puzzles, train_solutions)
    val_data = Dataset_4x4(val_puzzles, val_solutions)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)
    
    # Create HRM model
    model = HRM_4x4(hidden_dim=hidden_dim, max_iterations=max_iterations).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Max iterations: {max_iterations}\n")
    
    best_acc = 0
    Path('model').mkdir(exist_ok=True)
    
    # Training metrics
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'avg_iterations': [],
        'avg_residuals': []
    }
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        epoch_iterations = []
        epoch_residuals = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for puzzles, cells, digits in pbar:
            puzzles = puzzles.to(device)
            cells = cells.to(device).long()
            digits = digits.to(device).long()
            
            optimizer.zero_grad()
            
            # Forward pass with HRM
            cell_logits, digit_logits, traces = model(puzzles, return_traces=False)
            
            # Task loss: predict correct cell and digit
            task_loss = (nn.functional.cross_entropy(cell_logits, cells) + 
                        2.0 * nn.functional.cross_entropy(digit_logits, digits))
            
            # Halting penalty: encourage efficient iteration count
            halt_penalty = model.get_halt_penalty(traces, target_iterations=5)
            
            # Total loss
            loss = task_loss + halt_weight * halt_penalty
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Track metrics
            epoch_iterations.append(traces['num_iterations'])
            if traces['residuals']:
                epoch_residuals.append(traces['residuals'][-1])
            
            # Accuracy
            cell_pred = torch.argmax(cell_logits, dim=1)
            digit_pred = torch.argmax(digit_logits, dim=1)
            correct += ((cell_pred == cells) & (digit_pred == digits)).sum().item()
            total += len(puzzles)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}', 
                'acc': f'{correct/total:.1%}',
                'iters': f'{traces["num_iterations"]}'
            })
        
        train_acc = correct / total
        avg_iters = np.mean(epoch_iterations)
        avg_res = np.mean(epoch_residuals) if epoch_residuals else 0
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_iterations = []
        
        with torch.no_grad():
            for puzzles, cells, digits in val_loader:
                puzzles = puzzles.to(device)
                cells = cells.to(device).long()
                digits = digits.to(device).long()
                
                cell_logits, digit_logits, traces = model(puzzles)
                val_iterations.append(traces['num_iterations'])
                
                cell_pred = torch.argmax(cell_logits, dim=1)
                digit_pred = torch.argmax(digit_logits, dim=1)
                val_correct += ((cell_pred == cells) & (digit_pred == digits)).sum().item()
                val_total += len(puzzles)
        
        val_acc = val_correct / val_total
        val_avg_iters = np.mean(val_iterations)
        
        # Store history
        training_history['train_loss'].append(total_loss / len(train_loader))
        training_history['train_acc'].append(train_acc)
        training_history['val_acc'].append(val_acc)
        training_history['avg_iterations'].append(avg_iters)
        training_history['avg_residuals'].append(avg_res)
        
        print(f"Epoch {epoch}:")
        print(f"  Train: Acc={train_acc:.1%}, Iters={avg_iters:.1f}, Residual={avg_res:.4f}")
        print(f"  Val:   Acc={val_acc:.1%}, Iters={val_avg_iters:.1f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'model/hrm_4x4.pt')
            print(f"  ✓ Saved (best: {best_acc:.1%})")
        print()
    
    # Save training history
    with open('model/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"Training complete! Best accuracy: {best_acc:.1%}")
    print(f"Training history saved to model/training_history.json")
    
    return model


def test_model():
    """Test the trained HRM model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = HRM_4x4(hidden_dim=64, max_iterations=10).to(device)
    model.load_state_dict(torch.load('model/hrm_4x4.pt', map_location=device))
    model.eval()
    
    print("\n" + "="*50)
    print("Testing HRM 4x4 Solver")
    print("="*50)
    
    from generator import generate_puzzle
    
    correct = 0
    total_iters = []
    total_residuals = []
    
    for i in range(10):
        puzzle, solution = generate_puzzle()
        puzzle_tensor = torch.from_numpy(puzzle).unsqueeze(0).long().to(device)
        
        with torch.no_grad():
            cell_logits, digit_logits, traces = model(puzzle_tensor, return_traces=True)
            cell_pred = torch.argmax(cell_logits[0]).item()
            digit_pred = torch.argmax(digit_logits[0]).item() + 1
        
        pred_row, pred_col = cell_pred // 4, cell_pred % 4
        
        # Find first empty cell
        empty = np.argwhere(puzzle == 0)
        target_row, target_col = empty[0]
        target_digit = solution[target_row, target_col]
        
        is_correct = (pred_row == target_row and pred_col == target_col and 
                     digit_pred == target_digit)
        
        if is_correct:
            correct += 1
            status = "✓"
        else:
            status = "✗"
        
        total_iters.append(traces['num_iterations'])
        total_residuals.append(traces['residuals'][-1])
        
        print(f"Puzzle {i+1}: {status} "
              f"Pred=({pred_row},{pred_col})={digit_pred}, "
              f"True=({target_row},{target_col})={target_digit} "
              f"[iters={traces['num_iterations']}, res={traces['residuals'][-1]:.4f}]")
    
    print(f"\nAccuracy: {correct}/10 = {correct*10}%")
    print(f"Avg iterations: {np.mean(total_iters):.1f}")
    print(f"Avg final residual: {np.mean(total_residuals):.4f}")


if __name__ == "__main__":
    print("="*50)
    print("HRM 4x4 Solver - Training")
    print("="*50)
    print()
    
    model = train_model(
        num_epochs=20, 
        train_size=1000,
        hidden_dim=64,
        max_iterations=10,
        halt_weight=0.1
    )
    
    test_model()