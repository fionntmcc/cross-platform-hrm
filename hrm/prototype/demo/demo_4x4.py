import torch
import numpy as np
from models.model_4x4_layers import HRM_4x4
from hrm.prototype.generators.generator_4x4 import generate_puzzle, print_puzzle


def demo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = HRM_4x4(hidden_dim=64, n_outer_cycles=5, n_inner_steps=10).to(device)
    
    try:
        model.load_state_dict(torch.load('model/hrm_4x4.pt', map_location=device))
    except FileNotFoundError:
        print("No trained model found. Run train.py first.")
        return
    
    model.eval()
    
    print("\nDemo: HRM Sudoku Solver\n")
    
    # Generate and solve puzzles
    for i in range(5):
        print(f"\n--- Puzzle {i+1} ---")
        
        puzzle, solution = generate_puzzle()
        print_puzzle(puzzle, "Input")
        
        puzzle_tensor = torch.from_numpy(puzzle).unsqueeze(0).long().to(device)
        
        with torch.no_grad():
            cell_logits, digit_logits, traces = model(puzzle_tensor)
            
            cell_pred = torch.argmax(cell_logits[0]).item()
            digit_pred = torch.argmax(digit_logits[0]).item() + 1
            
            # Confidence scores
            cell_conf = torch.softmax(cell_logits[0], dim=0)[cell_pred].item()
            digit_conf = torch.softmax(digit_logits[0], dim=0)[digit_pred-1].item()
        
        pred_row, pred_col = cell_pred // 4, cell_pred % 4
        
        # Find actual target
        empty = np.argwhere(puzzle == 0)
        target_row, target_col = empty[0]
        target_digit = solution[target_row, target_col]
        
        print(f"\nModel prediction:")
        print(f"  Cell: ({pred_row}, {pred_col}) [confidence: {cell_conf:.1%}]")
        print(f"  Digit: {digit_pred} [confidence: {digit_conf:.1%}]")
        
        print(f"\nCorrect answer:")
        print(f"  Cell: ({target_row}, {target_col})")
        print(f"  Digit: {target_digit}")
        
        is_correct = (pred_row == target_row and pred_col == target_col and 
                     digit_pred == target_digit)
        
        if is_correct:
            print("\nCorrect")
        else:
            print("\nIncorrect")
        
        print_puzzle(solution, "Solution")


if __name__ == "__main__":
    demo()
