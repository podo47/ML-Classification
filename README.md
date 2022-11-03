# ML-Classification

## Content
1.All codes used are in "code" file 
  * Linear Classifier from scratch : LC
  * Linear classifier with least-squared manner : LS
  * Voted perceptron
  * SVM (Hard margin) : HardSVM
  * SVM (Soft margin) : SoftSVM
  * SVM by sklearn : sklearn

2.Figures of performance with different C are in  "figure" file
  * data - Performance with different C_data
  
  * crx - Performance with different C_crx
  
3.Given datasets "data.csv" and "crx.csv" are im "Given dataset" file

## Result
### ACCURACY OF EACH METHOD
|      | LC     | LS     | VP     | SVM(Hard)  | SVM(Soft) | SVM(sklearn) |
|:----:|:------:|:------:|:------:|:----------:|:---------:|:------------:|
| data | 0.9156 | 0.9649 | 0.9104 | 1.0        | 0.9824    | 0.9591       |
| crx  | 0.6937 | 0.5345 | 0.6217 | 0.6018     | 0.8760    | 0.8827       |

### COMPARISON OF MARGIN
|      | LC (Scratch) | SVM (Hard) |
|:----:|:------------:|:----------:|
| data | 0.0001       | 4.1371e-05 |
| crx  | 0.0017       | 0.4905     |

### Effective weighting value C
! [image](https://github.com/podo47/ML-Classification/raw/main/figure/Performance with different C_data.png)

! [image](https://github.com/podo47/ML-Classification/raw/main/figure/Performance with different C_crx.png)
