from prediction import BoardPredictor
import sys
import chess.svg


def main():
    # Check for correct number of command-line arguments
    if len(sys.argv) < 3:
        print("Usage: main.py <model_file_path> <image_file_path>")
        sys.exit(1)

    # Command-line arguments
    model_file_path = sys.argv[1]
    image_file_path = sys.argv[2]

    # Instantiate the BoardPredictor with the model file path
    predictor = BoardPredictor(model_file_path)

    # Perform prediction
    predicted_board = predictor.predict(image_file_path)

    # Render the predicted board to an SVG file or display it
    board_svg = chess.svg.board(board=predicted_board)
    with open("predicted_board.svg", "w") as f:
        f.write(board_svg)

    # Optionally display the board SVG in the console (if your terminal supports it)
    # print(board_svg)

    print("Prediction complete. Check the 'predicted_board.svg' file.")


if __name__ == "__main__":
    main()
