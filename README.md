# fenify-3D

Fenify-3D is a specialized project leveraging AI, Machine Learning (ML), and deep learning technologies to transform real-world images of chess boards into corresponding Forsyth-Edwards Notation (FEN) strings.

![Alt text](readme-assets/image-001.png)

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Training](#training)
4. [Accuracy](#accuracy)
5. [Inference](#inference)

## Introduction 

The Forsyth-Edwards Notation (FEN) is a standard notation crucial for delineating specific board positions in a chess game. FEN encapsulates all essential information, enabling the recommencement of a game from a defined position.

Fenify-3D is meticulously engineered to analyze an image of a physical chess board and deliver a precise FEN string, illustrating the current state of the board. This facilitates the real-time transcription of chess games through video input in downstream applications.

Distinct from existing chess game recording systems utilizing image and video data, Fenify-3D presents an innovative approach. It operates without necessitating a "top-down" perspective of the board, and accomidates various angles, provided the board maintains reasonable visibility. This methodological refinement enhances the adaptability and applicability of the system in diverse real-world scenarios. 

## Dataset 

The training dataset was composed of two primary components: synthetic and crowdsourced data.  

### Synthetic

Initial work started with synthetic data to determine project viability.  

![Alt text](readme-assets/image-002.png)

#### Simulator

Synthetic data was generated using Unity, employing chess boards and pieces sourced from Unity's Asset Store. Eight distinctive board sets were utilized (LowRes, Staunton, Abstract, Carved, Design, Dual, Medieval, ModernArt), paired with five unique piece sets (LowRes, HighRes, Staunton, Carved, Dual). A normalization process was applied to the dimensions of each board and piece set, ensuring compatibility and interchangeability among different sets.

In the Unity environment, randomization techniques were applied to foster a robust diversity of data within the confines of the simulated environment. The key factors subject to randomization included:

- Board Set
- Piece Set (accompanied by size jitter)
- Camera attributes such as position, angle, and focal point
- Placement of individual pieces
- Lighting attributes such as intensity and direction
- Background color

#### Piece Distribution

An initial research focus was on optimizing the randomization of the chess board to achieve two primary objectives:
- Enable replication using a physical board and piece set
- Minimize class imbalance to facilitate training

Various piece distributions were explored and experimented with, as summarized below:

| Distribution Type                | Description                                                                                       |
|----------------------------------|---------------------------------------------------------------------------------------------------|
| HalfEmptyRandomDistribution      | Each square has a 50% probability of being empty or occupied by a random piece.                    |
| StartingPositionDistribution     | Pieces are distributed based on the starting position with random square assignments.              |
| CompetitionDistribution          | Boards are selected randomly from a collection of positions from over 7k+ competitive games (Elo > 2000).|

A "FullyRandomDistribution" approach was not pursued due to impractical levels of occlusion resulting from overlapping pieces.

The HalfEmptyRandomDistribution method was dismissed following attempts to recreate it physically. The need for multiple piece sets per board made it impractical for use with crowdsourced participants.

CompetitionDistribution introduced a systemic bias due to the historical positioning of pieces, impeding optimal learning of visual features, as confirmed through experimentation. Thus, it was reserved for fine-tuning and test sets at later stages.

StartingPositionDistribution, despite its inherent challenges such as class imbalance and a fixed piece set, was chosen due to its feasibility for crowdsourcing, as it only necessitates the use of the standard chess sets. Training experiments conducted on synthetic data, affirmed the viability of training using the StartingPositionDistribution.

#### Generation

![Alt text](readme-assets/video-002.gif)

A dedicated Unity application was developed for the generation process. This application was tasked with randomizing various aspects, including the board, pieces, and the environment, followed by capturing a screenshot and saving it to the filesystem. This process was repeated in iterations.

Alongside the images, bounding box coordinates of the chessboard were also saved. This additional data facilitates the cropping of images to focus on relevant visual information, ensuring that the analysis is confined to the essential regions of the images.

In total, the application generated a dataset consisting of 30,000 diverse images.

### Crowdsourced

![Alt text](readme-assets/image-003.png)

Following a series of initial training experiments that yielded positive results, the methodology employed in synthetic data collection was expanded to incorporate crowdsourcing via Mechanical Turk. A coherent set of instructions, complemented by a compact web application, was developed and refined through an iterative process of trial and error. This approach facilitated the engagement of a broader participant base, leveraging diverse environments to enhance and enrich the dataset beyond the limitations of purely synthetic generation techniques.

#### Data Augmentation

![Alt text](readme-assets/image-004.png)

Constructing the simulation tool was a time-intensive process. However, it enabled the generation of incremental board images in a matter of milliseconds without requiring manual intervention. Conversely, arranging each randomized real-world board was a meticulous task, demanding careful and deliberate effort.

To optimize the data obtained from each uniquely arranged board, a strategy was employed to capture images from all four sides of the board, yielding four distinct images per arrangement. In a subsequent phase during training, these images, along with their board positions, will be subjected to random horizontal flips. This technique aims to enhance the dataset, effectively producing eight varied images from a single arranged board, thereby maximizing the utility of each manual arrangement in enriching the dataset for model training.

#### Collection Task

![Alt text](readme-assets/image-005.png)

The data collection procedure was refined through an iterative process of trial and error, as the author engaged in hands-on data gathering. A web application, hosted on Google Cloud, was developed to facilitate this process, with a design optimized for mobile user accessibility and interaction.

The application interface was organized systematically: 
- At the top, a randomly generated chessboard was displayed, complemented by its rotational views, utilizing a UI powered by chessboard.js for enhanced visibility and user experience.
- Following this, a 3D visualization was presented, showcasing a randomized angle of the chessboard for users to replicate.
- The lower section of the application featured a form with fields for uploading up to four images, an ID field, and a submission button.

Upon submission, the collected image files were securely stored in Google Cloud Storage, awaiting subsequent review and utilization in the project’s data processing and analysis stages.

#### Mechanical Turk 

![Alt text](readme-assets/image-006.png)

Administering the data collection process via Mechanical Turk and ensuring the submission quality posed substantial challenges. 

A significant portion, approximately 75%, of submissions from MTurk workers consisted of readily available stock images of chess boards sourced online. Such submissions necessitated removal, and the responsible workers were subsequently blocked. Genuine contributions, originating from workers who diligently followed the instructions, were less frequent. These contributors often reached the preset individual limit of 50 submissions. Instances of workers circumventing this restriction through the use of multiple accounts were also observed and necessitated vigilant monitoring and blocking to maintain the integrity of the data collection process. The imposition of the submission limit aimed to foster data diversity, preventing the overrepresentation of data from a limited number of contributors.

On average, each board arrangement consumed about 3 minutes of effort, culminating in the approval of 9505 images, 2500 of which were contributed by the author. Considering that each arrangement yielded four images, the total human effort invested amounts to approximately 120 hours. Overall, contributions were received from 150 distinct workers, with a select few surpassing the 50-board submission threshold. The financial expenditure for the Mechanical Turk utilization in this project amounted to approximately $1550.

![Alt text](readme-assets/image-009.png)

The dataset exhibited a pronounced skewness towards contributions from the author and a select group of reliable data gatherers who had access to multiple environments. This concentration of data sources resulted in a high standard of piece placement accuracy; however, it also limited the diversity of the dataset. Notably, approximately a quarter of the data originated from a limited number of settings orchestrated by the author. This lack of variety could potentially affect the model's ability to generalize to a broader range of real-world conditions. Addressing this imbalance is critical for enhancing the robustness and reliability of the model across varied chessboard setups and environments.

#### Quality Assurance

Maintaining high data quality necessitated substantial quality assurance efforts. Fortunately, the majority of issues encountered were common and could be addressed through automated corrective measures.

One recurrent issue was the misidentification of kings and queens in the crowdsourced submissions. Workers who made this error typically did so consistently across all their submissions. To rectify this, an automated process was implemented that programmatically adjusted the board assignments, swapping the positions of the king and queen. This approach resolved the issue effectively without necessitating resubmission by the worker.

Ambiguities in piece colors, particularly in certain piece sets, also presented challenges. This was especially notable in glass sets, where distinguishing between colors was somewhat nuanced. To standardize the data, clear pieces were uniformly categorized as black, and frosted pieces as white, with programmatic adjustments made to correct any inconsistencies without the need for resubmission.

Incorrect rotation orders were another common discrepancy, leading to mismatches between images and their corresponding board assignments. Initially, such boards were disqualified. However, with the availability of a trained model, automated corrections were introduced to identify and adjust improper rotations, thereby preserving the integrity of the data.

Minor piece placement errors were tolerated to a degree, permitting a small margin of one or two misplaced squares per sixty-four. Boards exhibiting more than two placement errors were deemed unacceptable. A comprehensive review of every square in each image was impractical. Typically, validations focused on the back row, and if found accurate, the remainder of the board was assumed to be correct.

Upon the establishment of a sufficiently trained model, the entire dataset underwent a comprehensive review and correction process to identify and amend any lingering data quality issues. 

## Training

Previous open source board detection models preprocess the board by dividing it into 64 squares and classifying each square separately. Fenify-3D avoids this preprocessing and predicts all 64 squares directly from the model. While harder to train, this methodology enables much wider viewing angles and doesn't fail with occlusion.    

### Dataset and Augmentation

![Alt text](readme-assets/image-007.png)

The dataset for training includes 30,000 synthetic and 9,505 crowdsourced images.  The dataset was also augmented using color and geometric transformations during training.  Augmentations were limited by two primary factors.  Vertical flips, horizontal flips, and significant rotations weren't possible since it would materially affect the visual representation of the pieces.  Piece placement is relative to the viewing angle, not an absolute placement.  Horizontal flips were implemented by wrapping both the image and board transformations together.  Color augmentations were also limited to prevent pixel intensity inversion.  If light and dark colors are inverted, the piece color assignments would be incorrect.  

The following transforms were used during training:
```python
T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25)
T.RandomRotation(degrees=(-15,15))
T.RandomPerspective(distortion_scale=0.25)
T.Grayscale(num_output_channels=3)
FlipTransform()
```

### Model

Pytorch, TorchVision, and Pytorch Lightning were the frameworks used to train the model.  The model consisted of a EfficientNetV2_S image decoder and a linear output layer consisting of 819 outputs grouping sixty four squares with thirteen possible outputs.  The forward function reshapes the output head into 64x13 outputs and then performs softmax on the piece dimension yielding predictions for the entire board.

The EfficientNet series including EfficientNet_B4 and EfficientNetV2_S were the only model architectures that converged to an accurate result.  Popular model architectures such as Resnet50 never reached a high enough accuracy to be useful.   

```python
class Model(pl.LightningModule):

    def __init__(self, resnet=None, piece_sets=[PieceSet.Binary, PieceSet.Colors, PieceSet.ColorBlind, PieceSet.Full],
                 lr=0.0001):
        super().__init__()
        self.piece_sets = piece_sets
        self.lr = lr
        self.resnet = resnet
        self.outputs = nn.Linear(1000, 64*13)
        self.losses = nn.ModuleList()
        for pc in piece_sets:
            loss = nn.CrossEntropyLoss(weight=pc.weights())
            self.losses.append(loss)

    def forward(self, x):
        x = F.relu(self.resnet(x))
        x = self.outputs(x)
        x = torch.reshape(x, (x.shape[0], 64, 13))
        return F.softmax(x,dim=2)
```

### Loss Function

Initial attempts to train a model using a standard class prediction per square failed because of the steep learning curve of matching visual features to exact board position.  In order to smooth out the learning gradient, piece sets were added to give the model "partial credit".  For example if the model predicts a white queen with 100% confidence instead of a white king with the Full piece set, it would experience the full loss.  With the Binary and Colors piece sets, it would experience no loss indicating a correct response.  

Training eventually converged using the Binary, Colors, ColorBlind, and Full piece sets. Ultimately the correct class is needed, but by creating a smoother learning gradient the model latched on to import features by solving easier versions of the board prediction problem.  


| Piece Set                | Descriptor |
|---------------------|------------|
| Binary              | empty / piece |
| Colors              | empty / white / black |
| MajorMinor          | empty / white minor / white major / black minor / black major |
| RoyalMajorMinor     | empty / white minor / white major / white royal / black minor / black major / black royal |
| ColorBlind          | empty / pawn / knight / bishop / rook / queen / king |
| Full                | empty / white pawn / white knight / white bishop / white rook / white queen / white king / black pawn / black knight / black bishop / black rook / black queen / black king |

### Hyperparameters and Scheduling

![Alt text](readme-assets/image-008.png)

The training and validation sets were 90% and 10% of the 39,505 combined synthetic and crowdsourced datasets.  A batch size of 16 images with a learning rate of 1e-4 was used for training for the first ~60 epochs which took around 10 hours. 

Training used learning rate and batch size annealing and gradually increased during both learning rate and batch size during the training process to improve the robustness and generalization of the model.  This schedule wasn't systematically tested but worked well in practice.  

## Accuracy

### Validation Set

#### Gross Metrics

The validation set comes from the same data distribution as the training set.  The following is the gross metrics from the validation set:

| Metric                  | Value                 |
|-------------------------|-----------------------|
| Loss                    | 3.88     |
| Binary Accuracy         | 0.999    |
| Colors Accuracy         | 0.974    |
| Color-blind Accuracy    | 0.974    |
| *Full Accuracy*         | *0.950*    |

#### Prediction Visualization

The graphic below is a visual example of four randomly selected images from the validation set including the raw image, the board's actual or correct position, the predicted position, and the full piece accuracy.    An X on the predicted column indicates an inaccurate square prediction.  

![Alt text](readme-assets/image-010.png)

The model shows strong predition performance by only predicting one to three squares incorrectly per board. 

#### Confusion Matrix 

The analysis of the confusion matrices highlights the model's predictive strengths and areas where further improvement could be beneficial.

*Colors Confusion Matrix*
![Alt text](readme-assets/image-012.png)

The confusion matrix for color prediction demonstrates a high level of accuracy in distinguishing between binary and colored piece sets. It reveals a minor tendency for the model to overpredict the presence of white pieces when they are actually black.

*Color Blind Confusion Matrix*
![Alt text](readme-assets/image-013.png) 

For the color blind confusion matrix, a logarithmic scale is employed to enhance visibility of problematic prediction areas. The first square, representing correct predictions for empty squares, is omitted to accentuate the predictive performance for actual pieces. The matrix indicates a notable challenge in correctly identifying pawns, with a higher rate of false negatives observed. Knights and bishops are frequently misclassified as pawns, likely due to visual similarities in certain perspectives and piece designs.

*Full Confusion Matrix*
![Alt text](readme-assets/image-014.png)

Similar to the color blind matrix, the full confusion matrix employs a logarithmic scale to better illustrate areas of misprediction. Regions without confusion are left unmarked. The most significant issue arises with the misidentification of black pawns, as indicated by the rate of false negatives. It is also observed that white pieces are occasionally mistaken for their black counterparts, which is reflected by a line parallel to the identity diagonal, representing correct predictions.

#### Analysis 

The model demonstrates commendable performance on StartingPositionDistribution randomized boards, achieving, on average, a per-square accuracy of 95%, with typically 3.2 squares misclassified out of 64.

Errors in predictions are generally confined to either piece type or color, rather than both. Interestingly, the model exhibits similar accuracy rates (97.4%) for predicting both piece type and color, despite piece prediction appearing more challenging. This could suggest an underlying issue, potentially arising from competing loss functions in the 'Colors' and 'ColorBlind' piece sets. Although prior ablation studies suggested the 'ColorBlind' piece set contributed value to the model's performance, this assumption may require reassessment given the current evidence.

Utilizing true random piece arrangements eliminates the model's ability to leverage common chess positions to improve its visual recognition capabilities, enforcing a stringent standard that each prediction must rely solely on visual cues without the benefit of statistical shortcuts. However, it appears that the model may be utilizing the fixed piece count inherent to the data collection design, as this is reflected in its performance on the test set.

This assessment indicates that while the model is performing well, there is room for further refinement.

### Test Set

The test set comes from a different data distribution than the training set.  It is composed of videos of actual games. The following is the gross metrics from the test set:

| Metric                        | Value   |
|-------------------------------|---------|
| Loss                          | 4.191   |
| Binary Accuracy               | 0.905   |
| Colors Accuracy               | 0.887   |
| Color-Blind Accuracy          | 0.864   |
| *Full Accuracy*                 | *0.852*   |

#### Prediction Visualization

The graphic below is a visual example of four randomly selected images from the test set including the raw image, the board's actual or correct position, the predicted position, and the full piece accuracy.  An X on the predicted column indicates an inaccurate square prediction. 

![Alt text](readme-assets/image-016.png)

The model performs quite poorly compared to the model on the validation set.  

#### Confusion Matrix 

The analysis of the confusion matrices highlights the model's predictive strengths and areas where further improvement could be beneficial.

*Colors Confusion Matrix*
![Alt text](readme-assets/image-015.png)

The confusion matrix for color prediction in the test set shows the deep accuracy flaws in the model for the test set.  The model significantly over predicts pieces on empty squares.

#### Analysis 

The included visualization delineates the prediction challenges encountered by the model on the test dataset:

![Alt text](readme-assets/image-017.png)

The model's accuracy improves with an increased number of pieces on the board. This trend suggests a systematic bias stemming from differences between the training/validation sets and the test set, with the model demonstrating a propensity to predict board occupancy close to 50% even in the absence of pieces.

To enhance the model's predictive performance on the test dataset, an additional phase of training is warranted. Leveraging a fine-tuning set comprised of actual game positions—akin to the approach utilized in 'fenify'—should yield improvements. Such a dataset would enable the model to harness statistical piece correlations to refine its predictions.

Given the relative simplicity of predicting empty squares, a modest training epoch count on a dataset comprising a few thousand images should suffice to mitigate the model’s bias toward specific piece counts.

## Inference

TODO