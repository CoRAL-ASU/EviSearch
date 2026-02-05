"""
Column mapping for clinical trial data visualization.
Maps CSV column names to display categories in exact order.
"""

# Category definitions with columns in exact order
COLUMN_CATEGORIES = {
    "📄 Trial Characteristics": [
        # Identifiers
        "NCT",
        "PubMed ID",
        "Trial Name",
        "Author",
        "Year",
        "Full Pub or Abstract",
        
        # Design
        "Phase",
        "Original/Follow Up",
        "Number of Arms Included",
        
        # Treatment Arms
        "Treatment Arm(s)",
        "Class of Agent in Treatment Arm 1",
        "Treatment Arm 1 Regimen",
        "Treatment Arm - N",
        
        # Control Arms
        "Control Arm",
        "Control Arm - N",
        
        # Overall
        "Total Participants - N",
        
        # Exposure Duration
        "Median Follow-Up Duration (mo)",
        "Median On-Treatment Duration (mo) | Treatment",
        "Median On-Treatment Duration (mo) | Control",
        
        # Quality of Life
        "Quality of Life reported",
        "Quality of Life Scale",
        
        # Reporting by Prognostic Groups
        "Reporting by prognostic groups - Y/N | Synchronous",
        "Reporting by prognostic groups - Y/N | Metachronous",
        "Reporting by prognostic groups - Y/N | High volume",
        "Reporting by prognostic groups - Y/N | Low volume",
        
        # Additional
        "Add-on Treatment",
        "Treatment Class",
        "Trial",
        "Type of Therapy",
        "COE_RCT_IND_OVERALL_RJ",
    ],
    
    "👥 Population Characteristics": [
        # Mode of Metastases
        "Mode of metastases - N (%) | Synchronous | Treatment",
        "Mode of metastases - N (%) | Synchronous | Control",
        "Mode of metastases - N (%) | Metachronous | Treatment",
        "Mode of metastases - N (%) | Metachronous | Control",
        
        # Volume of Disease
        "Volume of disease - N (%) | High | Treatment",
        "Volume of disease - N (%) | High | Control",
        "Volume of disease - N (%) | Low | Control",
        
        # Docetaxel Administration
        "Docetaxel administration - N (%) | Treatment",
        "Docetaxel administration - N (%) | Control",
        
        # Median Age
        "Median Age (years) | Treatment",
        "Median Age (years) | Control",
        
        # Race
        "Race - N (%) | White | Treatment",
        "Race - N (%) | White | Control",
        "Race - N (%) | Black or African American | Treatment",
        "Race - N (%) | Black or African American | Control",
        "Race - N (%) | Asian | Treatment",
        "Race - N (%) | Asian | Control",
        "Race - N (%) | Nat. Hawaiian or Pac. Islander | Treatment",
        "Race - N (%) | Nat. Hawaiian or Pac. Islander | Control",
        "Race - N (%) | Amer. Indian or Alaska Nat. | Treatment",
        "Race - N (%) | Amer. Indian or Alaska Nat. | Control",
        "Race - N (%) | Other | Treatment",
        "Race - N (%) | Other | Control",
        "Race - N (%) | Unknown | Treatment",
        "Race - N (%) | Unknown | Control",
        
        # Region
        "Region - N (%) | North America | Treatment",
        "Region - N (%) | North America | Control",
        "Region - N (%) | South America | Treatment",
        "Region - N (%) | South America | Control",
        "Region - N (%) | Europe | Treatment",
        "Region - N (%) | Europe | Control",
        "Region - N (%) | Africa | Treatment",
        "Region - N (%) | Africa | Control",
        "Region - N (%) | Asia | Treatment",
        "Region - N (%) | Asia | Control",
        "Region - N (%) | Oceania | Treatment",
        "Region - N (%) | Oceania | Control",
        
        # Performance Status
        "PS - N (%) | 0 | Treatment",
        "PS - N (%) | 0 | Control",
        "PS - N (%) | 1-2 | Treatment",
        "PS - N (%) | 1-2 | Control",
        
        # Gleason Score
        "Gleason score - N (%) | ≤ 7 | Treatment",
        "Gleason score - N (%) | ≤ 7 | Control",
        "Gleason score - N (%) | ≥ 8 | Treatment",
        "Gleason score - N (%) | ≥ 8 | Control",
        
        # Metastases
        "Metastases - N (%) | Liver | Treatment",
        "Metastases - N (%) | Liver | Control",
        "Metastases - N (%) | Lungs | Treatment",
        "Metastases - N (%) | Lungs | Control",
        "Metastases - N (%) | Bone | Treatment",
        "Metastases - N (%) | Bone | Control",
        "Metastases - N (%) | Nodal | Treatment",
        "Metastases - N (%) | Nodal | Control",
        
        # Previous Local Therapy
        "Previous local therapy - N (%) | Prostatectomy | Treatment",
        "Previous local therapy - N (%) | Prostatectomy | Control",
        "Previous local therapy - N (%) | Orchiectomy | Treatment",
        "Previous local therapy - N (%) | Orchiectomy | Control",
        "Previous local therapy - N (%) | Radiotherapy | Treatment",
        "Previous local therapy - N (%) | Radiotherapy | Control",
    ],
    
    "📊 Results for Overall Population": [
        # Endpoints
        "Primary Endpoint(s)",
        "Secondary Endpoint(s)",
        
        # Objective Response Rate
        "ORR - N (%) | Treatment | Overall",
        "ORR - N (%) | Treatment | CR",
        "ORR - N (%) | Treatment | SD",
        "ORR - N (%) | Treatment | PD",
        "ORR - N (%) | Control | Overall",
        "ORR - N (%) | Control | CR",
        "ORR - N (%) | Control | SD",
        "ORR - N (%) | Control | PD",
        
        # Adverse Events
        "Adverse Events - N (%) | All-Cause Grade 3 or Higher | Treatment",
        "Adverse Events - N (%) | All-Cause Grade 3 or Higher | Control",
        "Adverse Events - N (%) | Treatment-related Grade 3 or Higher | Treatment",
        "Adverse Events - N (%) | Treatment-related Grade 3 or Higher | Control",
        "Adverse Events - N (%) | Treatment-related Grade 5 | Treatment",
        "Adverse Events - N (%) | Treatment-related Grade 5 | Control",
        
        # Deaths
        "No. of Deaths - N | Treatment",
        "No. of Deaths - N | Control",
        
        # TTPSA
        "TTPSA (mo) | Treatment",
        "TTPSA (mo) | Control",
        
        # Overall Survival - OS Rate
        "OS Rate (%) | Overall | Treatment",
        "OS Rate (%) | Overall | Control",
        
        # Overall Survival - Median OS
        "Median OS (mo) | Overall | Treatment",
        "Median OS (mo) | Overall | Control",
        
        # Progression Free Survival - PFS Rate
        "PFS Rate (%) | Overall | Treatment",
        "PFS Rate (%) | Overall | Control",
        
        # Progression Free Survival - Median PFS
        "Median PFS (mo) | Overall | Treatment",
        "Median PFS (mo) | Overall | Control",
    ],
    
    "🎯 Results by Prognostic Groups": [
        # OS Rate by Groups
        "OS Rate (%) | Synchronous | Treatment",
        "OS Rate (%) | Synchronous | Control",
        "OS Rate (%) | Metachronous | Treatment",
        "OS Rate (%) | Metachronous | Control",
        "OS Rate (%) | High volume | Treatment",
        "OS Rate (%) | High volume | Control",
        "OS Rate (%) | Low volume | Treatment",
        "OS Rate (%) | Low volume | Control",
        
        # Median OS by Volume
        "Median OS (mo) | High volume | Treatment",
        "Median OS (mo) | High volume | Control",
        "Median OS (mo) | Low volume | Treatment",
        "Median OS (mo) | Low volume | Control",
        
        # Median PFS by Volume
        "Median PFS (mo) | High volume | Treatment",
        "Median PFS (mo) | High volume | Control",
        "Median PFS (mo) | Low volume | Treatment",
        "Median PFS (mo) | Low volume | Control",
    ],
}


# Reverse mapping: column -> category (for quick lookup)
COLUMN_TO_CATEGORY = {}
for category, columns in COLUMN_CATEGORIES.items():
    for col in columns:
        COLUMN_TO_CATEGORY[col] = category


def get_category_for_column(column_name):
    """Get the category for a given column name."""
    return COLUMN_TO_CATEGORY.get(column_name, "📋 Other")


def get_ordered_categories():
    """Get categories in display order."""
    return [
        "📄 Trial Characteristics",
        "👥 Population Characteristics",
        "📊 Results for Overall Population",
        "🎯 Results by Prognostic Groups",
    ]
