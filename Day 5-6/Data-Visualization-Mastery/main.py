"""
Data Visualization Mastery Project
Main entry point for the project
"""

def main():
    print("Data Visualization Mastery Project")
    print("Choose an option:")
    print("1. Run Visualization Gallery")
    print("2. Run Titanic EDA")
    print("3. Exit")

    choice = input("Enter your choice (1-3): ")

    if choice == "1":
        from visualization_gallery import main as viz_main
        viz_main()
    elif choice == "2":
        from Titanic_EDA import main as eda_main
        eda_main()
    elif choice == "3":
        print("Exiting...")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
