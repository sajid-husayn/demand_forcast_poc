import pandas as pd
import pymongo
from datetime import datetime
import numpy as np
import os

# MongoDB Configuration
MONGO_URI = "mongodb+srv://syedhussain:OZHk1GTYKSUBvIUM@cluster0.cwo0vp8.mongodb.net/"
DATABASE_NAME = "taniaWaters"
COLLECTION_NAME = "orderData"

# Excel file path (assuming it's in the same folder)
EXCEL_PATH = "Sales Copy.xlsx"

def clean_column_names(columns):
    """Clean and standardize column names to camelCase"""
    # Reference column names from first sheet - converted to camelCase
    reference_columns = [
        'billT', 'description', 'billingDate', 'salesOffice', 'salesOfficeDescription',
        'distributionChannel', 'dChannelName', 'material', 'itemDescription',
        'billedQuantity', 'su', 'billDoc', 'createdOn', 'soldTo', 'soldToPartyName', 'payer'
    ]
    
    # Mapping from original names to camelCase
    column_mapping = {
        'BillT': 'billT',
        'Description': 'description',
        'Billing Date': 'billingDate',
        'Sales Office': 'salesOffice',
        'Sales Office Description': 'salesOfficeDescription',
        'Distribution Channel': 'distributionChannel',
        'D Channel Name': 'dChannelName',
        'Material': 'material',
        'Item Description': 'itemDescription',
        'Billed Quantity': 'billedQuantity',
        'SU': 'su',
        'Bill. Doc.': 'billDoc',
        'Created On': 'createdOn',
        'Sold-to': 'soldTo',
        'Sold To Party Name': 'soldToPartyName',
        'Payer': 'payer'
    }
    
    cleaned = []
    for col in columns:
        col_clean = str(col).strip()
        # Convert to camelCase using mapping
        camel_case_col = column_mapping.get(col_clean, col_clean.replace(' ', '').replace('.', '').replace('-', ''))
        cleaned.append(camel_case_col)
    
    return cleaned

def process_excel_data():
    """Read Excel file and process all sheets"""
    try:
        print(f"Reading Excel file: {EXCEL_PATH}")
        
        # Read all sheets
        all_sheets = pd.read_excel(EXCEL_PATH, sheet_name=None)
        print(f"Found {len(all_sheets)} sheets: {list(all_sheets.keys())}")
        
        # Get reference columns from first sheet (in camelCase)
        first_sheet_name = list(all_sheets.keys())[0]
        first_sheet = all_sheets[first_sheet_name]
        reference_columns = clean_column_names(first_sheet.columns)
        
        # Define camelCase reference columns
        camel_case_reference = [
            'billT', 'description', 'billingDate', 'salesOffice', 'salesOfficeDescription',
            'distributionChannel', 'dChannelName', 'material', 'itemDescription',
            'billedQuantity', 'su', 'billDoc', 'createdOn', 'soldTo', 'soldToPartyName', 'payer'
        ]
        
        print(f"Reference columns from '{first_sheet_name}' (camelCase): {camel_case_reference}")
        
        combined_data = []
        
        for sheet_name, sheet_data in all_sheets.items():
            print(f"\nProcessing sheet: {sheet_name}")
            print(f"Original shape: {sheet_data.shape}")
            
            # Clean column names to camelCase
            sheet_data.columns = clean_column_names(sheet_data.columns)
            
            # Add missing columns with NaN values
            for ref_col in camel_case_reference:
                if ref_col not in sheet_data.columns:
                    sheet_data[ref_col] = np.nan
                    print(f"Added missing column: {ref_col}")
            
            # Reorder columns to match reference
            sheet_data = sheet_data.reindex(columns=camel_case_reference)
            
            # Add metadata in camelCase
            sheet_data['sourceSheet'] = sheet_name
            sheet_data['uploadTimestamp'] = datetime.now()
            
            # Convert to records
            records = sheet_data.to_dict('records')
            
            # Clean records (convert NaN to None for MongoDB)
            cleaned_records = []
            for record in records:
                cleaned_record = {}
                for key, value in record.items():
                    if pd.isna(value):
                        cleaned_record[key] = None
                    elif isinstance(value, (np.integer, np.floating)):
                        cleaned_record[key] = float(value) if not np.isnan(value) else None
                    else:
                        cleaned_record[key] = value
                cleaned_records.append(cleaned_record)
            
            combined_data.extend(cleaned_records)
            print(f"Added {len(cleaned_records)} records from {sheet_name}")
        
        print(f"\nTotal records to upload: {len(combined_data)}")
        return combined_data
        
    except Exception as e:
        print(f"Error processing Excel file: {str(e)}")
        return None

def upload_to_mongodb(data):
    """Upload data to MongoDB"""
    try:
        print(f"\nConnecting to MongoDB...")
        client = pymongo.MongoClient(MONGO_URI)
        
        # Test connection
        client.admin.command('ping')
        print("Successfully connected to MongoDB!")
        
        # Get database and collection
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]
        
        print(f"Database: {DATABASE_NAME}")
        print(f"Collection: {COLLECTION_NAME}")
        
        # Check if collection already has data
        existing_count = collection.count_documents({})
        print(f"Existing documents in collection: {existing_count}")
        
        if existing_count > 0:
            response = input("Collection already has data. Clear it first? (y/n): ")
            if response.lower() == 'y':
                result = collection.delete_many({})
                print(f"Deleted {result.deleted_count} existing documents")
        
        # Insert data in batches
        batch_size = 1000
        total_inserted = 0
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            result = collection.insert_many(batch)
            total_inserted += len(result.inserted_ids)
            print(f"Inserted batch {i//batch_size + 1}: {len(batch)} documents")
        
        print(f"\nSuccessfully uploaded {total_inserted} documents!")
        
        # Verify upload
        final_count = collection.count_documents({})
        print(f"Final document count in collection: {final_count}")
        
        # Show sample document
        sample = collection.find_one()
        if sample:
            print(f"\nSample document:")
            for key, value in list(sample.items())[:5]:
                print(f"  {key}: {value}")
            print("  ...")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"Error uploading to MongoDB: {str(e)}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("Excel to MongoDB Upload Script")
    print("=" * 60)
    
    # Check if Excel file exists
    if not os.path.exists(EXCEL_PATH):
        print(f"Error: Excel file '{EXCEL_PATH}' not found!")
        print("Make sure the file is in the same folder as this script.")
        return
    
    # Process Excel data
    data = process_excel_data()
    if data is None:
        print("Failed to process Excel data. Exiting.")
        return
    
    # Upload to MongoDB
    success = upload_to_mongodb(data)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Upload completed successfully!")
        print("✅ Your forecasting API can now use the MongoDB data")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Upload failed!")
        print("=" * 60)

if __name__ == "__main__":
    # Install required packages if not present
    try:
        import pymongo
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages:")
        print("pip install pymongo pandas numpy openpyxl")
        exit(1)
    
    main()
