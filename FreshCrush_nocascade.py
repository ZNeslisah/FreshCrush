from PyQt6.QtWidgets import QApplication, QMainWindow, QGridLayout, QPushButton, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon
import subprocess
import config
import random
import os
from PyQt6.QtWidgets import QSpacerItem, QSizePolicy, QFrame
from PyQt6.QtCore import QTimer


class CandyCrushGame(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Candy Crush Game")
        self.setFixedSize(QSize(1100, 990))
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        
        self.image_folder = None
        self.level_selection_ui()

        self.rows, self.cols = 14, 14  # Enlarged grid
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.image_folder = None 
    
    def clear_layout(self):
        """Clears the current layout to switch UI elements."""
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def level_selection_ui(self):
        """Creates an improved level selection UI with a modern layout."""
        self.clear_layout()
        
        self.main_layout.setContentsMargins(0, 220, 0, 0)  # Add top margin
        
        # Title Label
        label = QLabel("Select a Level")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 28px; font-weight: bold; color: #728FCE;")
        # Reduced top spacer (smaller value pushes label down)
        self.main_layout.addSpacerItem(QSpacerItem(80, 160, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        self.main_layout.addWidget(label)

        # Create a frame to group buttons
        button_frame = QFrame()
        button_frame.setStyleSheet("background-color: white; border-radius: 20px; padding: 20px;")
        frame_layout = QVBoxLayout(button_frame)
        frame_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Grid Layout for buttons
        button_layout = QGridLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frame_layout.addLayout(button_layout)

        self.levels = {
            "🍎🍌🍊🍆🥒 Beginner": config.fruitCrash_level1,
            "🍎🍌🍊🍆🥒🍍🥕🍄 Pro": config.fruitCrash_level2
        }

        row, col = 0, 0
        for level, folder in self.levels.items():
            button = QPushButton(level)
            button.setStyleSheet(
                "font-size: 18px; padding: 15px; min-width: 150px; font-weight: bold; "
                "border-radius: 10px; background-color: #4863A0; color: white;"
            )

            # Set icons if available
            button.setIconSize(QSize(40, 40))

            button.clicked.connect(lambda _, folder=folder: self.start_game(folder))
            button_layout.addWidget(button, row, col)

            col += 1
            if col > 1:  # Arrange in 2 columns
                col = 0
                row += 1

        self.main_layout.addWidget(button_frame)

        # Bottom spacer
        self.main_layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

    def start_game(self, image_folder):
        """Starts the game with the selected image folder."""
        self.image_folder = image_folder
        self.init_game_ui()


    def getCandies(self):
        """Randomly selects image files for the grid and processes them with ANN."""

        # Get all image files in the folder
        all_files = [f for f in os.listdir(self.image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        if not all_files:
            raise ValueError("No image files found in the dataset folder!")

        # Randomly assign initial file names
        file_grid = [[random.choice(all_files) for _ in range(self.cols)] for _ in range(self.rows)]
        self.selected_files = [file for row in file_grid for file in row]  # Flatten the list

        # Write selected file names to `selected_files.txt`
        selected_files_path = f'{self.current_dir}/selected_files.txt'
        with open(selected_files_path, "w") as f:
            for file in self.selected_files:
                f.write(file + "\n")

        # Call the ANN script to classify the selected images
        ann_command = [
            'python3',
            f'{self.current_dir}/noveltyDetection_fruitCrash.py',
            selected_files_path,
            f'{self.current_dir}/ann_output.txt',
            f'{self.image_folder}'
        ]
        subprocess.run(ann_command)

        # Read ANN output and store mappings (file_name → class_name)
        class_mapping = {}
        self.class_points = {}
        ann_output_path = f'{self.current_dir}/ann_output.txt'
        with open(ann_output_path, "r") as f:
            for line in f:
                clean_line = line.strip().replace("(", "").replace(")", "").replace("'", "")
                parts = clean_line.split(", ")
                if len(parts) == 2:
                    file_name, class_point = parts
                    parts = class_point.split(' ', 1)
                    if len(parts) > 1:
                        class_name, point = parts  # Split at the first space
                        point = float(point)  # Convert point to a number (assuming it's numerical)
                        class_mapping[file_name.strip()] = class_name.strip()
                        self.class_points[class_name.strip()] = point  # Store point in 
                        # print("Point:", point, "Class Name:", class_name)
                    else:
                        class_name = class_point.strip()
                        class_mapping[file_name.strip()] = class_name
                        self.class_points[class_name] = 1.0  
                        class_name = class_point

                    # class_mapping[file_name.strip()] = class_name.strip()
        # print("Class Mapping Dictionary:", class_mapping.values())
        def is_valid_placement(grid, r, c, class_name):
            """Check if placing `class_name` at (r, c) violates the constraints."""
            # Check horizontal
            if c >= 2 and grid[r][c - 1] == class_name and grid[r][c - 2] == class_name:
                return False
            # Check vertical
            if r >= 2 and grid[r - 1][c] == class_name and grid[r - 2][c] == class_name:
                return False
            return True

        # **Create an ordered list of available image files per class**
        class_image_pool = {}
        for file_name, class_name in class_mapping.items():
            if class_name not in class_image_pool:
                class_image_pool[class_name] = []
            class_image_pool[class_name].append(file_name)

        # **Index tracking for cyclic image selection**
        class_image_index = {class_name: 0 for class_name in class_image_pool}

        # **Ensure valid grid generation with correct file tracking**
        grid = [["" for _ in range(self.cols)] for _ in range(self.rows)]
        file_assignment = [["" for _ in range(self.cols)] for _ in range(self.rows)]

        for r in range(self.rows):
            for c in range(self.cols):
                valid_classes = [class_name for class_name in class_mapping.values() if is_valid_placement(grid, r, c, class_name)]
                # print("Valid Classes:", valid_classes)
                if valid_classes:
                    selected_class = random.choice(valid_classes)
                else:
                    selected_class = random.choice(list(class_mapping.values()))

                # **Assign class name to the grid**
                grid[r][c] = selected_class

                # **Cycle through available images for this class (Round-Robin)**
                if selected_class in class_image_pool and class_image_pool[selected_class]:
                    file_index = class_image_index[selected_class]
                    file_assignment[r][c] = class_image_pool[selected_class][file_index]

                    # Move to the next image in the list (circular rotation)
                    class_image_index[selected_class] = (file_index + 1) % len(class_image_pool[selected_class])

        # Create image mapping: (row, column) → file path
        # image_mapping = {(r, c): os.path.join(self.image_folder, file_assignment[r][c]) for r in range(self.rows) for c in range(self.cols)}
        if self.image_folder == config.fruitCrash_level1:
            image_mapping = {(r, c): os.path.join(config.fruitCrash_level1_original, file_assignment[r][c]) for r in range(self.rows) for c in range(self.cols)}
        elif self.image_folder == config.fruitCrash_level2:
            image_mapping = {(r, c): os.path.join(config.fruitCrash_level2_original, file_assignment[r][c]) for r in range(self.rows) for c in range(self.cols)}
        else:
            image_mapping = {(r, c): os.path.join(self.image_folder, file_assignment[r][c]) for r in range(self.rows) for c in range(self.cols)}



        # print("Validated Grid:")
        # for row in grid:
        #     print(row)

        return grid, image_mapping

 
    def has_match(self, grid, r, c):
        """Checks if placing a candy at (r, c) creates a match of 3 or more."""
        candy = grid[r][c]

        # Check horizontal matches
        if c >= 2 and grid[r][c - 1] == candy and grid[r][c - 2] == candy:
            return True
        if c <= self.cols - 3 and grid[r][c + 1] == candy and grid[r][c + 2] == candy:
            return True
        if c >= 1 and c <= self.cols - 2 and grid[r][c - 1] == candy and grid[r][c + 1] == candy:
            return True

        # Check vertical matches
        if r >= 2 and grid[r - 1][c] == candy and grid[r - 2][c] == candy:
            return True
        if r <= self.rows - 3 and grid[r + 1][c] == candy and grid[r + 2][c] == candy:
            return True
        if r >= 1 and r <= self.rows - 2 and grid[r - 1][c] == candy and grid[r + 1][c] == candy:
            return True

        return False

    def init_game_ui(self):
      
        """Initializes the game UI."""
        self.clear_layout()
        self.grid, self.image_mapping = self.getCandies()
        self.selected_buttons = []
        self.score = 0  # Initialize score
        
        # Set main layout spacing and margins to remove extra space
        self.main_layout.setSpacing(10)  # Removes default spacing between widgets
        self.main_layout.setContentsMargins(0, 10, 0, 0)  # Removes padding from all sides

        # Score Label Styling (Ensuring it appears at the top)
        self.score_label = QLabel(f"⭐ Score: {self.score} ⭐")
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #ff5733; padding: 5px; border-radius: 10px;"
            "background-color: #ffcccb; border: 2px solid #ff5733;"
        )
        self.score_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.main_layout.addWidget(self.score_label)

        self.grid_layout = QGridLayout()
        self.main_layout.addLayout(self.grid_layout)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.grid_layout.setSpacing(1)

        self.buttons = [[QPushButton(self.grid[r][c]) for c in range(self.cols)] for r in range(self.rows)]

        self.solve_button = QPushButton("Solve")
        self.solve_button.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.solve_button.clicked.connect(self.solve_move)
        self.main_layout.addWidget(self.solve_button)

        for r in range(self.rows):
            for c in range(self.cols):
                # Create a container widget to hold both the button and the label
                container = QWidget()
                vbox = QVBoxLayout(container)
                vbox.setContentsMargins(0, 0, 0, 0)  # Remove margins
                vbox.setSpacing(5)  # Adjust spacing between icon and text

                # Create the button
                button = QPushButton()
                button.setFixedSize(70, 70)  # Set button size
                icon_path = self.image_mapping[(r, c)]
                button.setIcon(QIcon(icon_path))
                button.setIconSize(QSize(50, 50))  # Adjust icon size
                
                # Correctly store the button reference
                self.buttons[r][c] = button  # Store reference to button
                
                button.clicked.connect(lambda _, row=r, col=c: self.handle_click(row, col))

                # Create the label for text
                label = QLabel(self.grid[r][c])  # Set the name as label
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center text
                label.setStyleSheet("font-size: 12px; font-weight: bold; color: black;")

                # Add button and label to the layout
                vbox.addWidget(button, alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
                vbox.addWidget(label, alignment=Qt.AlignmentFlag.AlignHCenter)

                # Add container to grid layout
                self.grid_layout.addWidget(container, r, c)



    def handle_click(self, row, col):
        """Handles button clicks to swap candies."""
        self.selected_buttons.append((row, col))

        if len(self.selected_buttons) == 1:
            self.buttons[row][col].setStyleSheet("background-color: yellow; font-size: 12px;")
        elif len(self.selected_buttons) == 2:
            pos1, pos2 = self.selected_buttons
            if abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) != 1:
                self.reset_selection()
                return

            self.swap_candies(pos1, pos2)
            self.update_buttons()

            matches = self.find_matches()
            if matches:
                self.score += self.calculate_score(matches)
                self.score_label.setText(f"Score: {self.score}")
                self.clear_matches(matches)
            else:
                self.swap_candies(pos2, pos1)  # Swap back
                self.update_buttons()

            self.reset_selection()

    def reset_selection(self):
        """Resets the selection of buttons."""
        for row, col in self.selected_buttons:
            self.buttons[row][col].setStyleSheet("font-size: 12px; width: 20px; height: 40px;")
        self.selected_buttons = []

    def swap_candies(self, pos1, pos2):
        """Swaps two candies in the grid and updates the image mapping."""
        r1, c1 = pos1
        r2, c2 = pos2

        # Swap in the logical grid
        self.grid[r1][c1], self.grid[r2][c2] = self.grid[r2][c2], self.grid[r1][c1]

        # Swap in the image mapping
        self.image_mapping[(r1, c1)], self.image_mapping[(r2, c2)] = (
            self.image_mapping[(r2, c2)],
            self.image_mapping[(r1, c1)],
        )


    def update_buttons(self):
        """Updates the button icons and labels to reflect the current grid."""
        for r in range(self.rows):
            for c in range(self.cols):
                container = self.grid_layout.itemAtPosition(r, c).widget()
                if container:
                    layout = container.layout()
                    if layout:
                        for i in range(layout.count()):
                            item = layout.itemAt(i)
                            widget = item.widget()
                            
                            if isinstance(widget, QPushButton):  # Update icon
                                if self.grid[r][c] == " ":
                                    widget.setIcon(QIcon())  # Remove the icon
                                else:
                                    icon_path = self.image_mapping[(r, c)]
                                    widget.setIcon(QIcon(icon_path))
                                    widget.setIconSize(QSize(50, 50))  # Maintain consistent icon size
                            
                            elif isinstance(widget, QLabel):  # Update text
                                if self.grid[r][c] == " ":
                                    widget.setText("")  # Remove text
                                else:
                                    widget.setText(self.grid[r][c])  # Update label text

    def find_matches(self):
        """Finds horizontal and vertical matches in the grid."""
        matches = set()

        # Check horizontal matches
        for r in range(self.rows):
            for c in range(self.cols - 2):
                if self.grid[r][c] == self.grid[r][c + 1] == self.grid[r][c + 2] and self.grid[r][c] != " ":
                    matches.add((r, c))
                    matches.add((r, c + 1))
                    matches.add((r, c + 2))

        # Check vertical matches
        for c in range(self.cols):
            for r in range(self.rows - 2):
                if self.grid[r][c] == self.grid[r + 1][c] == self.grid[r + 2][c] and self.grid[r][c] != " ":
                    matches.add((r, c))
                    matches.add((r + 1, c))
                    matches.add((r + 2, c))

        return matches


    def clear_matches(self, matches):
        """Clears matched candies and updates the grid visually."""
        if not matches:
            return

        for r, c in matches:
            self.grid[r][c] = " "  # Empty the grid slot
            
            # Retrieve the container widget that holds the button and label
            container = self.grid_layout.itemAtPosition(r, c).widget()
            if container:
                layout = container.layout()
                if layout:
                    # Find the QPushButton and QLabel inside the layout
                    for i in range(layout.count()):
                        item = layout.itemAt(i)
                        if isinstance(item.widget(), QPushButton):
                            item.widget().setIcon(QIcon())  # Remove the icon
                        elif isinstance(item.widget(), QLabel):
                            item.widget().setText("")  # Remove the text

    def calculate_score(self, matches):
        """Calculates the score based on the number of matches and their class points."""
        total_score = len(matches)
        if len(matches) > 0:
            for r, c in matches:
                class_name = self.grid[r][c]
            class_point = self.class_points.get(class_name, 1.0)  # Default to 1.0 if class not found
            total_score *= class_point  # Multiply each match by its point value

        return int(total_score)  # Convert to integer for clean scoring

    
    def solve(self):
        """Finds the best move based on the highest score gained from matching."""
        best_move = None
        max_score = 0  # Track the highest score gained

        for r in range(self.rows):
            for c in range(self.cols):
                # Check right swap
                if c < self.cols - 1:
                    self.swap_candies((r, c), (r, c + 1))
                    matches = self.find_matches()
                    score = self.calculate_score(matches)  # Use class point multipliers
                    
                    if score > max_score:
                        max_score = score
                        best_move = ((r, c), (r, c + 1))
                    
                    self.swap_candies((r, c + 1), (r, c))  # Swap back

                # Check down swap
                if r < self.rows - 1:
                    self.swap_candies((r, c), (r + 1, c))
                    matches = self.find_matches()
                    score = self.calculate_score(matches)  # Use class point multipliers
                    
                    if score > max_score:
                        max_score = score
                        best_move = ((r, c), (r + 1, c))
                    
                    self.swap_candies((r + 1, c), (r, c))  # Swap back

        return best_move

    

    def solve_move(self):
        """Automatically solves all possible moves when the solve button is clicked."""
        def solve_step():
            move = self.solve()
            if move:
                pos1, pos2 = move
                self.swap_candies(pos1, pos2)
                self.update_buttons()
                matches = self.find_matches()

                if matches:
                    self.score += self.calculate_score(matches)
                    self.score_label.setText(f"Score: {self.score}")
                    self.clear_matches(matches)
                    self.update_buttons()
                    QTimer.singleShot(100, solve_step)  # Delay for UI update and call solve_step again
                else:
                    self.swap_candies(pos2, pos1)  # Swap back if no match
                    self.update_buttons()
            else:
                self.score_label.setText(f"No valid moves! Score: {self.score}")

        solve_step()

if __name__ == "__main__":
    app = QApplication([])
    game = CandyCrushGame()
    game.show()
    app.exec()


































# import random
# from PyQt6.QtWidgets import QApplication, QMainWindow, QGridLayout, QPushButton, QWidget, QLabel, QVBoxLayout
# from PyQt6.QtCore import QSize, Qt
# import os
# import subprocess
# from PyQt6.QtGui import QIcon
# import time
# import config

# import random
# from PyQt6.QtWidgets import QApplication, QMainWindow, QGridLayout, QPushButton, QWidget, QLabel, QVBoxLayout, QHBoxLayout
# from PyQt6.QtCore import QSize, Qt
# import os
# import subprocess
# from PyQt6.QtGui import QIcon
# import config

# class LevelSelection(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Select Level")
#         self.setFixedSize(QSize(1000, 940))

#         central_widget = QWidget()
#         self.setCentralWidget(central_widget)

#         main_layout = QVBoxLayout()
#         central_widget.setLayout(main_layout)

#         label = QLabel("Select a Level")
#         label.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         label.setStyleSheet("font-size: 18px; font-weight: bold;")
#         main_layout.addWidget(label)

#         button_layout = QHBoxLayout()
#         main_layout.addLayout(button_layout)

#         self.levels = {
#             "Easy": config.fruitCrash_test_dir,
#             "Medium": config.fruitCrash_test_dir,
#             "Hard": config.fruitCrash_test_dir
#         }

#         for level, folder in self.levels.items():
#             button = QPushButton(level)
#             button.setStyleSheet("font-size: 14px; padding: 10px;")
#             button.clicked.connect(lambda _, folder=folder: self.start_game(folder))
#             button_layout.addWidget(button)

#     def start_game(self, image_folder):
#         self.game = CandyCrushGame(image_folder)
#         self.game.show()
#         self.close()


# class CandyCrushGame(QMainWindow):
#     def __init__(self, image_folder):
#         super().__init__()
#         self.rows, self.cols = 8, 8  # Enlarged grid
#         self.current_dir = os.path.dirname(os.path.abspath(__file__))
#         # self.image_folder = config.fruitCrash_test_dir 
#         self.image_folder = image_folder 

#         self.grid, self.image_mapping = self.getCandies()
#         self.selected_buttons = []
#         self.score = 0  # Initialize score
#         self.init_ui()

#     def getCandies(self):
#         """Randomly selects image files for the grid and processes them with ANN."""

#         # Get all image files in the folder
#         all_files = [f for f in os.listdir(self.image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

#         if not all_files:
#             raise ValueError("No image files found in the dataset folder!")

#         # Randomly assign initial file names
#         file_grid = [[random.choice(all_files) for _ in range(self.cols)] for _ in range(self.rows)]
#         self.selected_files = [file for row in file_grid for file in row]  # Flatten the list

#         # Write selected file names to `selected_files.txt`
#         selected_files_path = f'{self.current_dir}/selected_files.txt'
#         with open(selected_files_path, "w") as f:
#             for file in self.selected_files:
#                 f.write(file + "\n")

#         # Call the ANN script to classify the selected images
#         ann_command = [
#             'python3',
#             f'{self.current_dir}/noveltyDetection_fruitCrash.py',
#             selected_files_path,
#             f'{self.current_dir}/ann_output.txt'
#         ]
#         subprocess.run(ann_command)

#         # Read ANN output and store mappings (file_name → class_name)
#         class_mapping = {}
#         ann_output_path = f'{self.current_dir}/ann_output.txt'
#         with open(ann_output_path, "r") as f:
#             for line in f:
#                 clean_line = line.strip().replace("(", "").replace(")", "").replace("'", "")
#                 parts = clean_line.split(", ")
#                 if len(parts) == 2:
#                     file_name, class_name = parts
#                     # print("File Name:", file_name, "Class Name:", class_name)
#                     class_mapping[file_name.strip()] = class_name.strip()
#                     # print("Class Mapping:", class_mapping)

#         # print("Class Mapping Dictionary:", class_mapping.values())
#         def is_valid_placement(grid, r, c, class_name):
#             """Check if placing `class_name` at (r, c) violates the constraints."""
#             # Check horizontal
#             if c >= 2 and grid[r][c - 1] == class_name and grid[r][c - 2] == class_name:
#                 return False
#             # Check vertical
#             if r >= 2 and grid[r - 1][c] == class_name and grid[r - 2][c] == class_name:
#                 return False
#             return True

#         # **Create an ordered list of available image files per class**
#         class_image_pool = {}
#         for file_name, class_name in class_mapping.items():
#             if class_name not in class_image_pool:
#                 class_image_pool[class_name] = []
#             class_image_pool[class_name].append(file_name)

#         # **Index tracking for cyclic image selection**
#         class_image_index = {class_name: 0 for class_name in class_image_pool}

#         # **Ensure valid grid generation with correct file tracking**
#         grid = [["" for _ in range(self.cols)] for _ in range(self.rows)]
#         file_assignment = [["" for _ in range(self.cols)] for _ in range(self.rows)]

#         for r in range(self.rows):
#             for c in range(self.cols):
#                 valid_classes = [class_name for class_name in class_mapping.values() if is_valid_placement(grid, r, c, class_name)]
#                 # print("Valid Classes:", valid_classes)
#                 if valid_classes:
#                     selected_class = random.choice(valid_classes)
#                 else:
#                     selected_class = random.choice(list(class_mapping.values()))

#                 # **Assign class name to the grid**
#                 grid[r][c] = selected_class

#                 # **Cycle through available images for this class (Round-Robin)**
#                 if selected_class in class_image_pool and class_image_pool[selected_class]:
#                     file_index = class_image_index[selected_class]
#                     file_assignment[r][c] = class_image_pool[selected_class][file_index]

#                     # Move to the next image in the list (circular rotation)
#                     class_image_index[selected_class] = (file_index + 1) % len(class_image_pool[selected_class])

#         # Create image mapping: (row, column) → file path
#         image_mapping = {(r, c): os.path.join(self.image_folder, file_assignment[r][c]) for r in range(self.rows) for c in range(self.cols)}

#         print("Validated Grid:")
#         for row in grid:
#             print(row)

#         return grid, image_mapping

 
#     def has_match(self, grid, r, c):
#         """Checks if placing a candy at (r, c) creates a match of 3 or more."""
#         candy = grid[r][c]

#         # Check horizontal matches
#         if c >= 2 and grid[r][c - 1] == candy and grid[r][c - 2] == candy:
#             return True
#         if c <= self.cols - 3 and grid[r][c + 1] == candy and grid[r][c + 2] == candy:
#             return True
#         if c >= 1 and c <= self.cols - 2 and grid[r][c - 1] == candy and grid[r][c + 1] == candy:
#             return True

#         # Check vertical matches
#         if r >= 2 and grid[r - 1][c] == candy and grid[r - 2][c] == candy:
#             return True
#         if r <= self.rows - 3 and grid[r + 1][c] == candy and grid[r + 2][c] == candy:
#             return True
#         if r >= 1 and r <= self.rows - 2 and grid[r - 1][c] == candy and grid[r + 1][c] == candy:
#             return True

#         return False

#     def init_ui(self):
#         """Initializes the UI."""
#         self.setWindowTitle("Candy Crush Game")
#         self.setFixedSize(QSize(1000, 940))  # Adjusted size for enlarged grid

#         central_widget = QWidget()
#         self.setCentralWidget(central_widget)

#         main_layout = QVBoxLayout()
#         central_widget.setLayout(main_layout)

#         self.score_label = QLabel(f"Score: {self.score}")
#         self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
#         self.score_label.setStyleSheet("font-size: 18px; font-weight: bold;")
#         main_layout.addWidget(self.score_label)

#         self.grid_layout = QGridLayout()
#         main_layout.addLayout(self.grid_layout)

#         self.buttons = [[QPushButton(self.grid[r][c]) for c in range(self.cols)] for r in range(self.rows)]
#         for r in range(self.rows):
#             for c in range(self.cols):
#                 button = self.buttons[r][c]
#                 button.setFixedSize(120, 120)  # Fix the button size
#                 icon_path = self.image_mapping[(r,c)]
#                 button.setIcon(QIcon(icon_path))
#                 button.setIconSize(QSize(60, 60))  
#                 button.setStyleSheet("font-size: 12px; width: 20px; height: 40px;")  # Enlarged icons
#                 button.clicked.connect(lambda _, row=r, col=c: self.handle_click(row, col))
#                 self.grid_layout.addWidget(button, r, c)

#     def handle_click(self, row, col):
#         """Handles button clicks to swap candies."""
#         self.selected_buttons.append((row, col))

#         if len(self.selected_buttons) == 1:
#             self.buttons[row][col].setStyleSheet("background-color: yellow; font-size: 12px;")
#         elif len(self.selected_buttons) == 2:
#             pos1, pos2 = self.selected_buttons
#             if abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) != 1:
#                 self.reset_selection()
#                 return

#             self.swap_candies(pos1, pos2)
#             self.update_buttons()

#             matches = self.find_matches()
#             if matches:
#                 self.score += self.calculate_score(matches)
#                 self.score_label.setText(f"Score: {self.score}")
#                 self.clear_matches(matches)
#             else:
#                 self.swap_candies(pos2, pos1)  # Swap back
#                 self.update_buttons()

#             self.reset_selection()

#     def reset_selection(self):
#         """Resets the selection of buttons."""
#         for row, col in self.selected_buttons:
#             self.buttons[row][col].setStyleSheet("font-size: 12px; width: 20px; height: 40px;")
#         self.selected_buttons = []

#     def swap_candies(self, pos1, pos2):
#         """Swaps two candies in the grid and updates the image mapping."""
#         r1, c1 = pos1
#         r2, c2 = pos2

#         # Swap in the logical grid
#         self.grid[r1][c1], self.grid[r2][c2] = self.grid[r2][c2], self.grid[r1][c1]

#         # Swap in the image mapping
#         self.image_mapping[(r1, c1)], self.image_mapping[(r2, c2)] = (
#             self.image_mapping[(r2, c2)],
#             self.image_mapping[(r1, c1)],
#         )

#     def update_buttons(self):
#         """Updates the button icons to reflect the current grid."""
#         for r in range(self.rows):
#             for c in range(self.cols):
#                 if self.grid[r][c] == " ":
#                     self.buttons[r][c].setIcon(QIcon())  # Set empty icon for cleared cells
#                     self.buttons[r][c].setText("")  # Empty text
#                     self.buttons[r][c].setStyleSheet("background-color: white;")  # Optional: blank background
#                 else:
#                     icon_path = self.image_mapping[(r,c)]
#                     self.buttons[r][c].setIcon(QIcon(icon_path))
#                     self.buttons[r][c].setIconSize(QSize(50, 50))  # Maintain consistent icon size
#                     self.buttons[r][c].setText(self.grid[r][c])
#                     self.buttons[r][c].setStyleSheet("font-size: 12px; width: 20px; height: 40px;")


#     def find_matches(self):
#         """Finds horizontal and vertical matches in the grid."""
#         matches = set()

#         # Check horizontal matches
#         for r in range(self.rows):
#             for c in range(self.cols - 2):
#                 if self.grid[r][c] == self.grid[r][c + 1] == self.grid[r][c + 2] and self.grid[r][c] != " ":
#                     matches.add((r, c))
#                     matches.add((r, c + 1))
#                     matches.add((r, c + 2))

#         # Check vertical matches
#         for c in range(self.cols):
#             for r in range(self.rows - 2):
#                 if self.grid[r][c] == self.grid[r + 1][c] == self.grid[r + 2][c] and self.grid[r][c] != " ":
#                     matches.add((r, c))
#                     matches.add((r + 1, c))
#                     matches.add((r + 2, c))

#         return matches


#     def clear_matches(self, matches):
#         """Clears matched candies and updates the grid visually."""
#         if not matches:
#             return

#         for r, c in matches:
#             self.grid[r][c] = " "
#             self.buttons[r][c].setIcon(QIcon())  # Remove the icon
#             self.buttons[r][c].setText(" ")
#             self.buttons[r][c].setIcon(QIcon())  # Remove the icon


#     def calculate_score(self, matches):
#         """Calculates the score based on the number of matches."""
#         return len(matches)

# # if __name__ == "__main__":
# #     app = QApplication([])
# #     game = CandyCrushGame()
# #     game.show()
# #     app.exec()

# if __name__ == "__main__":
#     app = QApplication([])
#     level_selection = LevelSelection()
#     level_selection.show()
#     app.exec()






























