# Image Classification Feature Guide

## Overview
This feature allows you to test the **real impact of label flipping attacks** on federated learning models by uploading images and seeing how the trained model classifies them.

## How It Works

### 1. Run an Experiment
First, run a federated learning experiment with your chosen configuration:

#### Clean Experiment (No Attack)
```
- Poisoned Workers: 0
- Attack Method: Any (won't matter)
- Selection Strategy: RandomSelectionStrategy
- Quick Mode: ✓ (for faster results)
```
**Expected Result:** Model trains properly and achieves good accuracy (~85%+)

#### Poisoned Experiment (With Attack)
```
- Poisoned Workers: 10-25
- Attack Method: replace_1_with_9 (or any other)
- Selection Strategy: RandomSelectionStrategy
- Quick Mode: ✓
```
**Expected Result:** Model accuracy degrades due to poisoned training data

### 2. View Results
After the experiment completes, go to the **Image Testing** tab

### 3. Upload an Image
- Click "Select Image" and upload any image
- The system will automatically:
  - Resize the image to the correct dimensions (28x28 for Fashion-MNIST, 32x32 for CIFAR-10)
  - Convert to the correct color format
  - Normalize pixel values
  - Run inference using THIS experiment's trained model

### 4. See the Impact
Compare results between clean and poisoned experiments:

#### Clean Model Results (0 poisoned workers):
```
Predicted: T-shirt/top (87.3% confidence)
All other classes: <5% each
```

#### Poisoned Model Results (20 poisoned workers):
```
Predicted: Wrong class or random distribution
All classes: ~10% each (random guessing)
OR
Predicted: Flipped class (e.g., Trouser instead of T-shirt)
```

## Key Features

### 1. Experiment-Specific Models
- Each experiment saves its own trained model
- Classification uses the ACTUAL model from that specific experiment
- No simulation or fake results

### 2. Attack Impact Demonstration
- Clean experiments show accurate classification
- Poisoned experiments show degraded performance
- Direct evidence of how attacks affect model quality

### 3. Real-Time Feedback
- Upload any image
- Get instant classification results
- See confidence scores for all classes

## Supported Datasets

### Fashion-MNIST
- **Classes:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- **Image Format:** 28x28 grayscale
- **Best Images:** Clothing items on white/plain background

### CIFAR-10
- **Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- **Image Format:** 32x32 RGB
- **Best Images:** Clear objects, centered composition

## Usage Tips

### For Best Results:
1. **Use high-quality images** with clear subjects
2. **Test the same image** on both clean and poisoned models to compare
3. **Run multiple experiments** with different attack configurations
4. **Use Quick Mode** for faster iteration during testing

### Understanding Results:
- **High confidence (>70%)**: Model is confident in its prediction
- **Low confidence (<30%)**: Model is uncertain (sign of attack impact)
- **Uniform distribution (~10% each)**: Model is completely broken (severe attack)

## Example Workflow

### Demonstrating Attack Impact:

#### Step 1: Create Baseline (Clean Model)
```
1. Set Poisoned Workers = 0
2. Run experiment
3. Wait for completion
4. Go to Image Testing tab
5. Upload a bird image
6. Result: "bird (89.2% confidence)" ✓
```

#### Step 2: Create Attacked Model
```
1. Set Poisoned Workers = 20
2. Set Attack = replace_2_with_7 (bird → horse)
3. Run experiment
4. Wait for completion
5. Go to Image Testing tab
6. Upload the SAME bird image
7. Result: "horse (45.3% confidence)" or random distribution ✗
```

#### Step 3: Compare
Now you have clear evidence that the attack degraded the model's ability to classify correctly!

## Technical Details

### Model Training
- **Quick Mode:** 10 epochs (~2-3 minutes)
- **Full Mode:** 50 epochs (~10-15 minutes)
- Uses actual federated averaging algorithm
- Real PyTorch training with backpropagation

### Model Storage
- Models saved as: `results/{experiment_id}_final_model.pth`
- Contains complete model state dictionary
- Can be reloaded for future classification

### Image Preprocessing
```python
# Fashion-MNIST
1. Convert to grayscale
2. Resize to 28x28
3. Normalize: pixel / 255.0

# CIFAR-10
1. Convert to RGB
2. Resize to 32x32
3. Normalize: (pixel - mean) / std
```

## Troubleshooting

### "Model not found for experiment"
- Make sure the experiment has completed successfully
- Check that the experiment ran in REAL mode (not simulation)
- Verify the model file exists: `results/{exp_id}_final_model.pth`

### "Classification failed"
- Ensure image is a valid format (JPG, PNG, etc.)
- Try a different image
- Check that the experiment completed without errors

### Results seem random (all ~10%)
- This might be EXPECTED if the experiment had heavy poisoning
- Try running a clean experiment (0 poisoned workers) to compare
- If clean model also shows random results, check training configuration

## API Endpoints

### Classify Image
```http
POST /api/classify/{experiment_id}
Content-Type: multipart/form-data

file: <image_file>
```

**Response:**
```json
{
  "success": true,
  "result": {
    "predicted_class": 2,
    "confidence": 0.873,
    "all_probabilities": [0.03, 0.05, 0.873, ...],
    "class_names": ["T-shirt/top", "Trouser", ...],
    "dataset_type": "fashion_mnist"
  }
}
```

## Future Enhancements
- Side-by-side comparison view
- Batch image testing
- Confusion matrix visualization
- Attack success rate metrics
- Export classification results to CSV
