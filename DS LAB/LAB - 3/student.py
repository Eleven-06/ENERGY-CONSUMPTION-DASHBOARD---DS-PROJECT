import pandas as pd

# Create a dictionary of student data
student_data = {
    "Roll No": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    "Name": [
        "Aditya", "Rahul", "Priya", "Sneha", "Arjun",
        "Kiran", "Anjali", "Rohit", "Meena", "Vikram"
    ],
    "Age": [20, 21, 20, 19, 21, 20, 19, 22, 20, 21],
    "Section": ["A", "A", "B", "B", "A", "C", "C", "B", "A", "C"],
    "Maths": [85, 78, 92, 88, 75, 90, 81, 69, 95, 84],
    "Physics": [80, 74, 89, 90, 77, 85, 79, 72, 93, 82],
    "Chemistry": [82, 76, 91, 87, 79, 88, 84, 70, 94, 86]
}

# Create DataFrame
df = pd.DataFrame(student_data)

# Display DataFrame
print("Student DataFrame:")
print(df)

# Save DataFrame to CSV file
df.to_csv("students.csv", index=False)

print("\nCSV file 'students.csv' created successfully!")
