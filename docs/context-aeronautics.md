                                                                               
  The Core Problem                                                              
                                                                                
  CFD software (ANSYS, OpenFOAM, etc.) is expensive and computationally         
  intensive. AI can act as a surrogate model - trained on physics simulation    
  data to provide fast, accurate predictions.                                   
                                                                                
  Key Design Decisions                                                          
                                                                                
  1. Airfoil Representation (Critical Choice)                                   
  Method: NACA parameters                                                       
  Pros: Simple, well-known                                                      
  Cons: Limited to NACA family                                                  
  ────────────────────────────────────────                                      
  Method: Raw coordinates                                                       
  Pros: Exact representation                                                    
  Cons: Variable length, not smooth                                             
  ────────────────────────────────────────                                      
  Method: CST (Class Shape Transform)                                           
  Pros: General, smooth, ~12-20 params                                          
  Cons: Slightly complex math                                                   
  ────────────────────────────────────────                                      
  Method: Bezier curves                                                         
  Pros: Flexible                                                                
  Cons: Hard to enforce constraints                                             
  My recommendation: CST parameterization - Used by Boeing/NASA, can represent  
  virtually any airfoil, mathematically smooth, easy to constrain.              
                                                                                
  2. Data Generation Strategy                                                   
                                                                                
  - XFOIL (panel method): Free, validated, fast enough for large datasets       
  - Can generate Cl, Cd, Cm, Cp distributions across Re and AoA ranges          
  - Synthetic data is valid here because XFOIL is physics-based and             
  well-validated                                                                
                                                                                
  3. What to Predict                                                            
                                                                                
  - Primary: Cl (lift), Cd (drag), Cl/Cd (efficiency)                           
  - Secondary: Cp distribution (pressure), transition point                     
  - Inputs: CST parameters + Reynolds number + Angle of Attack                  
                                                                                
  4. Quality Gates (Non-Negotiable for Engineering)                             
                                                                                
  ┌─────────────────────────────────────────────────────────┐                   
  │  ENGINEERING QUALITY PIPELINE                          │                    
  ├─────────────────────────────────────────────────────────┤                   
  │  1. Physical Plausibility     │ Cd > 0, |Cl| < 3, etc. │                    
  │  2. Uncertainty Quantification│ Ensemble or MC Dropout │                    
  │  3. OOD Detection             │ Reject extrapolations  │                    
  │  4. Validation Cases          │ Known airfoils (NACA)  │                    
  │  5. Operating Envelope        │ Re, AoA range checks   │                    
  │  6. Sensitivity Analysis      │ Input perturbation     │                    
  └─────────────────────────────────────────────────────────┘                   
                                                                                
  5. Proposed Architecture                                                      
                                                                                
  Input: CST params (12) + Re + AoA → [Feature Engineering]                     
                                            ↓                                   
                                [Ensemble Neural Network]                       
                                            ↓                                   
                      ┌─────────────────────┴─────────────────────┐             
                      ↓                                           ↓             
              [Coefficient Head]                         [Distribution Head]    
              Cl, Cd, Cm ± uncertainty                   Cp(x) via 1D decoder   
                      ↓                                           ↓             
              [Physical Constraints]                    [Integration Check]     
              (bounds, monotonicity)                    (Cl from Cp matches?)   
                      ↓                                                         
              [Confidence Score + OOD Flag]                                     
                                                                                
  My Recommendation                                                             
                                                                                
  Start with the most validated, general approach:                              
                                                                                
  1. CST-parameterized airfoils (general representation)                        
  2. XFOIL-generated dataset (physics-validated synthetic data)                 
  3. Ensemble of MLPs for uncertainty (simpler than Bayesian, production-ready) 
  4. Strict validation suite against NACA 0012, 2412, 4415 (extensively studied)
                                                                                
  This gives us a surrogate model that can predict aerodynamic coefficients in  
  milliseconds instead of minutes, with quantified uncertainty.                 
                                                                                
  Let me set up the project structure with proper engineering rigor.