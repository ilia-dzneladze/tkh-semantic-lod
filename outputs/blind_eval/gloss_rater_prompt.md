You are the rater in a blind evaluation. Each item below has a one-sentence
gloss, written to describe a group of terms, and a list of terms. The list
is a sample of one group of terms from a knowledge graph built from
machine-learning and materials-science research papers. It may or may not
be the group the gloss was written for. Rate whether the listed terms
support the gloss:

  accurate: the listed terms clearly support what the gloss says, and it
            fits most of them.
  vague:    nothing in the gloss is contradicted, but it is too general to
            tell this list apart from many others, or it fits only a small
            part of the list.
  wrong:    the gloss says something the listed terms contradict, or it
            does not fit most of them.

Judge only by what the listed terms support, not by what you know about
the field. Do not use any tools, files or searches. Everything you need is
in this message. Reply with ONLY a JSON object mapping each item id to one
of "accurate", "vague" or "wrong", for example {"G001": "vague"}, covering
all 72 items.

G001
  gloss: SOAP kernel variants and two-/three-body structural descriptors used to represent atomic environments while preserving symmetry.
  terms:
    - Three‑body angular module
    - CFID descriptor set
    - Three-body descriptor
    - Hierarchical descriptor set
    - 2-body descriptor
    - Findsym symmetry analysis
    - Chemical element permutation
G002
  gloss: A broad, heterogeneous list of cited DFT functionals, codes, and ML-potential methods (PBE, CP2K, AFLOW, sGDML, etc.) without one unifying theme.
  terms:
    - PBE functional
    - BAML
    - ECFP4
    - ACE
    - FCHL revisited
    - DeepH
    - Magpie
    - CPMD
    - CDVAE
    - Because the ACE basis is complete and hierarchical, it can in principle achieve arbitrary accuracy for any scalar, vectorial or tensorial property.
    - Magpie features
    - ANAKIN-ME
    - Roost
    - PBEsol
    - RMM-DIIS
G003
  gloss: Wide-ranging claims and methods on machine-learned interatomic potentials (GAP, ACE, moment tensor potentials) and their accuracy relative to DFT for silicon and other systems.
  terms:
    - GRU
    - HDAD
    - DFT+U
    - GW@LDA+U
    - LDA
    - The AFLOW standard enables reproducible high-throughput DFT calculations through standardized parameter sets for k-point density, energy cut-offs, exchange-correlation functionals, pseudopotentials, DFT+U parameters, and convergence criteria.
    - AFLOW implements DFT+U using the Dudarev formalism with default Ueff values for a wide range of elements.
    - GGA+U
    - Benchmarking beyond‑DFT methods
    - Computational efficiency vs. performance trade-off
    - Self-interaction error in DFT
    - Computational cost in high-throughput DFT
    - System size limitations of DFT
    - High‑throughput DFT property calculation
    - GRU update
G004
  gloss: Techniques for active-learning data acquisition, transfer learning, and training-set cleaning/curation used to build ML potential training workflows.
  terms:
    - Active learning workflow for sampling free‑energy derivative data
    - Multi‑modal training data support
    - Representative environment selection
    - Iterative self-consistent training protocol
    - uncorrelated snapshots selection
    - Iterative training loop workflow
    - Training set cleaning procedure
    - Domain shift between training and evaluation
    - efficient training set construction
    - Transfer‑learning for small‑data properties
    - Zero-shot transfer
    - double precision training
    - iterative retraining cycle
    - Self-consistent training
    - Iterative variable selection
G005
  gloss: Problems, metrics, and techniques around GPU acceleration, parallel scaling, and numerical precision trade-offs for training and running ML potentials.
  terms:
    - Methfessel-Paxton
    - MD-17 Ethanol
    - MD-17 Malonaldehyde
    - MD-17 Salicylic acid
    - MD-17 Toluene
    - MD-17 Uracil
    - CCSD(T) Molecular Set – Malonaldehyde
    - Malonaldehyde
    - Methane
    - Dipeptides
    - SPICE
    - Quench configurations
    - MD-17 Azobenzene (original)
    - MD-17 Aspirin (revised)
    - MD-17 Malonaldehyde (revised)
G006
  gloss: Techniques for fine-tuning and zero-shot transfer of pretrained models (GNoME, Wren), applied to tasks like molecular vibrational spectra prediction.
  terms:
    - GNoME
    - Fine-tuning models pre-trained on OMat24 to the MAD dataset achieves near-converged validation metrics in 30-100 epochs versus 700+ epochs when training from scratch, halving validation energy and force errors and achieving 50% lower test error than training on MAD alone.
    - Fine‑tuning an energy model on a benzene vibrational spectrum improves spectrum predictions for unseen molecules such as toluene and phenol.
    - Data download utilities
    - GNoME
    - MAD dataset fine-tuning
    - MPtrj dataset
    - Togo Database
    - Single-element dataset (Zuo et al.)
    - MPtrj dataset
    - relaxation on OC20 dataset
    - Fine-tuning
    - Fine‑tuning with experimental vibrational spectra
    - fine-tuning on bespoke datasets
    - Multi-dataset aggregation
G007
  gloss: Tasks, methods, and datasets for predicting melting temperatures, solid-solid phase transitions, and phase diagrams using phase-field and ML approaches.
  terms:
    - Assuming the order‑disorder interface energy is half the anti‑phase boundary energy, the interface energy employed in phase‑field simulations is approximately 15.45 mJ m⁻².
    - Polar‑nonpolar and polar‑polar transitions are more numerous than nonpolar‑nonpolar ones, with 214 polar‑nonpolar, 173 polar‑polar and 228 nonpolar‑nonpolar stable transitions predicted in the 300 K–600 K range.
    - The current screening neglects thermal‑expansion effects, which may cause discrepancies with experimental transition temperatures, especially when expansion strongly influences phase stability.
    - NVT thermostat during training
    - Quench configurations
    - Binary Ionic Compounds Melting Temperature Dataset
    - Structural phase transformations
    - phase diagram prediction
    - Martensitic phase‑transition free‑energy calculation in iron
    - Thermodynamic stability classification
    - Thermoelectric performance prediction
    - Materials thermodynamic stability prediction
    - melting temperature prediction for binary ionic solids
    - Debye temperature estimation
    - Nosé-Hoover thermostat
G008
  gloss: A varied set of atomistic/molecular-dynamics simulation tasks and methods (VASP, GAP, training protocols) addressing the accuracy-cost tradeoff versus first-principles MD.
  terms:
    - The Materials Project aims to reduce the typical 10–20 year timeline for advanced material development by providing large‑scale accurate information, thereby accelerating discovery for clean energy, electronics, and other sectors.
    - Future development plans include adding calculations of surface energies, elastic constants, point defects, finite‑temperature properties, and implementing a two‑way ‘calculations‑on‑demand’ feature for user‑submitted compounds.
    - AFLOW can automatically calculate a wide suite of physical observables across a large database of structures with minimal human intervention.
    - AFLOW can automatically create nanoparticle structures of arbitrary radius and separation, enabling high‑throughput nanoparticle studies.
    - The AFLOW database currently contains approximately 400 experimental prototype structures and several million bcc/fcc/hcp‑derived superstructures.
    - Materials Project database
    - Rapid-prototyping of new materials in silico
    - Cost-effective, data-driven materials design
    - Open-source platforms for robust, sophisticated materials analyses
    - materials-by-design applications
    - Transfer learning of element embeddings from large‑scale formation‑energy models to improve predictions of band gaps and elastic moduli
    - Strategies to mitigate data limitations in materials‑science machine learning
    - Slow, trial‑and‑error materials discovery
    - computational materials science
    - computational materials design
G009
  gloss: A large, varied set of tasks and problems for predicting materials properties (elastic moduli, energies, forces) via ML models like JARVIS-ML and Matbench.
  terms:
    - Elasticity
    - Evaluating new materials ML algorithms on the Matbench benchmark
    - Accelerated property prediction
    - Materials property prediction benchmarking
    - Crystal property prediction
    - Elastic constant prediction
    - Dielectric and piezoelectric response prediction
    - Materials property prediction via ML
    - Machine‑learning property prediction
    - Bulk modulus prediction
    - Metallicity classification
    - Bulk metallic glass formation prediction
    - Steel yield strength regression
    - machine learning for materials
    - Mechanical properties calculation
G010
  gloss: JARVIS-related claims and tasks around energy-conserving force prediction, force-field benchmarking, and scaling of ML potential accuracy with dataset size.
  terms:
    - Conjugate gradients
    - JARVIS‑FF automatically computes bulk modulus, defect formation energies, and phonon spectra for each force‑field and compares them with DFT results to assess force‑field quality.
    - Training SchNet with forces on only 1,000 examples yields better accuracy than training on 50,000 examples using energies alone, demonstrating that force information improves data efficiency.
    - The GNN’s stress predictions are substantially more accurate than on‑the‑fly stress calculations from the force field, which exhibit higher errors.
    - virial in loss function
    - MSE loss
    - Energy‑conserving force field learning
    - Accurate, fast energy/force/stress prediction
    - Force prediction
    - Force‑error benchmarking
    - Atomic von Mises stress prediction
    - DFT energy and force prediction
    - ML method benchmarking
    - machine learning benchmarking
    - Energy‑force‑virial joint loss
G011
  gloss: Claims, methods, and problems around Deep Potential (DeePMD) and SchNet-style neural network potentials for molecular dynamics of small molecules and water.
  terms:
    - Deep Potential MD (water)
    - NVNMD (non-von Neumann molecular dynamics) is 50x-100x faster than regular MD simulations.
    - DeePMD-kit has added MPI implementation for multi-device training and MD simulations using Horovod for data-parallel distributed training.
    - Pre-relaxing hypothetical structures with U-MLIP before ab initio calculations resulted in approximately 3× time savings compared to running ab initio on un-relaxed structures.
    - MLIP MD would unlock gains of orders of magnitude in accessible length and time scales versus AIMD, enabling study of new physical regimes.
    - DeePMD accurately reproduces radial distribution functions and three‑body angular distribution functions of liquid water and ice, matching AIMD results.
    - Integration with i‑PI for NPT simulations
    - Ice Ih
    - Ice Ih (b)
    - NequIP
    - Non-equilibrium Molecular Dynamics
    - Classical Force Field Molecular Dynamics
    - Molecular dynamics simulation
    - Classical molecular dynamics simulations
    - per-layer TorchScript files
G012
  gloss: A list of specific named neural network architectures (GemNet, DimeNet, MEGNet, PhysNet, SevenNet, etc.) for interatomic potentials and property prediction.
  terms:
    - LightGBM
    - DPA-1
    - e3nn
    - AFLOW
    - UFF
    - TPOT
    - JAX-MD
    - E3Linear layer
    - CFID
    - Aconvasp command‑line tool
    - GNoME models
    - MoS2
    - HIP-NN
    - TPOT
    - DP-GEN
G013
  gloss: Tasks and claims around screening and simulating solid-state electrolytes and cathode materials for lithium-ion batteries, including ionic diffusion prediction.
  terms:
    - Three‑body angular module
    - CFID descriptor set
    - Three-body descriptor
    - Hierarchical descriptor set
    - 2-body descriptor
    - Findsym symmetry analysis
    - Chemical element permutation
G014
  gloss: Methods and components for predicting DFT Hamiltonian matrices from atomic structure using graph neural networks, targeting band structure and polarizability.
  terms:
    - Within the training temperature range, LiFlow’s predicted diffusivity D* closely matches reference values from Winter and Gómez‑Bombarelli.
    - CHGNet enables nanosecond‑scale molecular dynamics simulations of garnet Li₃La₃Te₂O₁₂, yielding lithium self‑diffusion coefficients with quantified uncertainty.
    - Finite‑size tests using a larger supercell and an ML‑IAP (SevenNet) show that diffusion coefficients converge within statistical uncertainties, confirming that the AIMD results are not significantly affected by supercell size.
    - The DeePMD model successfully reproduces the superionic behavior of solid-state electrolytes with linear Lithium/Sodium mean squared displacement over time and diffusion coefficient uncertainty of about 20% or smaller, which is precise enough for battery applications.
    - diffusion pretraining scheme
    - Layered materials and solid-electrolyte candidates
    - Migration barrier prediction accuracy
    - High‑throughput screening of ionic conductivity
    - High‑throughput screening of Li‑ion battery anodes
    - Ionic conductor classification
    - Ion solution simulation
    - Li-ion conductor identification
    - Ionic diffusion coefficient prediction
    - Li diffusion simulation in LGPS
    - Minimum ion-ion distance cutoff
G015
  gloss: Tasks and claims on using a ConvLSTM model to predict crack patterns, fracture toughness, and Vicker's hardness from molecular dynamics data.
  terms:
    - The ConvLSTM model accurately predicts fracture patterns, crack‑length trends, and both mode‑I (tensile) and mode‑II (shear) loading responses across crystal orientations, closely matching atomistic molecular dynamics simulations, with only a minor discrepancy for the x100 orientation under mode‑II shear.
    - Vicker's hardness estimation model
    - Modified probabilistic substitution model
    - Combinatorial blow-up in candidate generation
    - Gradient‑orientation fracture prediction
    - One‑hot crack labeling
G016
  gloss: Optimization techniques (grid search, Bayesian optimization, evolutionary algorithms, gradient descent) used for hyperparameter tuning and materials design search.
  terms:
    - Generalized Gradient Approximation
    - Adaptive meshing and time stepping
    - Genetic‑algorithm‑driven inverse design
    - Harris-Foulkes force/stress corrector
    - Conjugate Gradients
    - Nonlinear parameter optimization
    - Gradient clipping
    - Hyperparameter tuning
    - Adaptive mesh refinement with hanging nodes
    - Bayesian optimization with surrogate model
    - Stochastic gradient descent on quadrature grids
    - Genetic‑algorithm optimization
    - Double grid technique
    - Random structure search
    - conjugate-gradient relaxation
G017
  gloss: Techniques for predicting convex-hull (thermodynamic stability) distance and for aggregating/padding neighbor lists in graph neural network potentials.
  terms:
    - Meta-learning allows ML algorithms to quickly learn and adapt to new tasks by leveraging experience from solving similar tasks with relatively small datasets.
    - Active learning workflow for sampling free‑energy derivative data
    - Multi‑modal training data support
    - Iterative self-consistent training protocol
    - uncorrelated snapshots selection
    - Training set cleaning procedure
    - Meta-learning
    - Multi-device scalability
    - Transfer‑learning for small‑data properties
    - Zero-shot transfer
    - iterative training
    - concurrent learning procedure
    - double precision training
    - dynamic load balancing
    - Iterative variable selection
G018
  gloss: Cited works, methods, and components centered on crystal graph neural network architectures (CGCNN and equivariant variants) for materials property prediction.
  terms:
    - Graph Network Simulator
    - Gated Graph Neural Network
    - Spectral Graph Convolution
    - Duvenaud et al. Graph Convolutional Networks
    - Graph Transformers
    - Bond graph
    - Graph neural network for structural models
    - apply the general MPNN approach to practically important graph problems
    - Materials Graph Network
    - Materials 3-body Graph Network
    - Graph Neural Network interatomic potentials
    - Three-body Materials Graph Network
    - Scalable high‑dimensional node representations
    - Graph convolution
    - Graph construction from atomic coordinates
G019
  gloss: Techniques for fine-tuning and zero-shot transfer of pretrained models (GNoME, Wren), applied to tasks like molecular vibrational spectra prediction.
  terms:
    - Phonax can predict molecular vibrational spectra and correctly assign IR/Raman active modes by analyzing irreducible representations.
    - Active learning workflow with Billiard Walk sampling
    - Prior selector model
    - JARVIS‑ML
    - AutoML stage
    - ElementProperty featurizer
    - DP operators library
    - Iterative training workflow
    - Power wall bottleneck
    - GPU memory scalability
    - Scaling model capacity without sacrificing speed
    - Experimental Plan Generation
    - Multi‑task Huber loss
    - Delta learning
    - Multi-property training
G020
  gloss: A large, diverse set of claims and tasks applying machine-learned potentials (GAP, Deep Potential, M3GNet, etc.) to varied materials simulations; no single tight theme.
  terms:
    - Symmetry-Preserving Inter-Atomic Potential
    - Brenner potential
    - The model predicts fracture toughness values that are comparable to those obtained from atomistic molecular dynamics simulations.
    - NequIP reproduces structural and kinetic properties from ab‑initio molecular dynamics simulations with high fidelity.
    - MEGNet models achieve lower mean absolute errors than SchNet on 11 of the 13 QM9 molecular properties evaluated.
    - Increasing the number of MEGNet blocks improves accuracy for properties that depend on longer‑range interactions, such as zero‑point vibrational energy, electronic spatial extent, and dipole moment.
    - GAP predicts diamond lattice parameter (3.539 Å) very close to DFT-LDA reference (3.532 Å), outperforming screened Tersoff potential (3.566 Å), with bulk modulus and elastic constants reasonably close to DFT reference values.
    - Sparse set of representative atoms
    - Stability‑ranking and heat‑map visualization
    - Liquid carbon 5000K
    - Frozen-framework potential energy surface (PES) descriptor screening
    - Tersoff potential
    - Bicrystal fracture prediction
    - Gap deformation potential prediction
    - High-temperature surface reconstruction simulations
G021
  gloss: JARVIS and AFLOW infrastructure, workflows, and tasks for electronic-structure, band-gap, and phonon property calculation and data access.
  terms:
    - Phonon DOS benchmark
    - SiO2 polymorph phonon calculations
    - Linear response phonon method
    - band gap (eV)
    - Automated phonon and vibrational property extraction
    - electronic structure
    - Phonon and elasticity calculations
    - phonon band structure calculation
    - hybrid functional band‑structure calculation
    - Band gap prediction
    - Band gap regression
    - Phonon dispersion calculation
    - Band structure calculation
    - electronic structure calculations
    - zero‑weight extra k‑path for band structures
G022
  gloss: Classical machine-learning techniques such as regression, clustering, and SHAP-based feature-importance analysis applied to materials property prediction.
  terms:
    - Symbolic Regression
    - Kernel Ridge Regression
    - Tree ensemble model
    - Cluster-specific regression models
    - SHAP feature importance analysis
    - Cluster-based regression
    - Feature standardization
    - Genetic‑algorithm selection of cluster‑expansion basis functions
    - Cluster expansion parameterization
    - Atomic Cluster Expansion
    - Regularized Gaussian‑process regression
    - Decision‑tree regression
    - Random Forest feature importance thresholding
    - Data cleaning and feature reduction
    - Composition-based hash filtering
G023
  gloss: Techniques for building test sets and cross-validation schemes (k-fold, leave-one-cluster-out, out-of-distribution splits) to evaluate materials ML models.
  terms:
    - OrbNet
    - ResNet
    - NewtonNet
    - Architecture modifications reduced SevenNet-0 total model parameters from 16.24 million (GNoME) to 0.84 million.
    - MEGNET blocks
    - MEGNetLayer
    - MEGNet
    - DimeNet
    - ænet
G024
  gloss: Technical components for radial-basis and cutoff-function descriptors, including Atomic Cluster Expansion (ACE) and its magnetic/spin-orbit extensions.
  terms:
    - Gaussian quadrature
    - The E(3)‑equivariant GNN surrogate provides superior accuracy compared with third‑ and fifth‑order Gaussian quadrature for thermal expansion of copper, martensitic phase transition of iron, and grain‑boundary energy calculations, matching molecular dynamics reference data.
    - Moments Tensor Potentials (MTP) and the Spectral Neighbor Analysis Potential (SNAP) can be expressed exactly as special cases of the ACE.
    - The ACE formulation is invariant under translation, rotation, inversion and permutation of identical atoms.
    - Learnable radial basis with Bessel functions
    - Atomic Cluster Expansion (ACE) high‑body‑order features
    - Irreducible Basis Set B
    - Linear Combination of Atomic Orbitals
    - Frame dependence of quadrature‑based energy averages
    - Energy discontinuities due to neighbor‑cutoff changes
    - Radial distribution function analysis
    - Change‑of‑basis using unitary P matrix
    - Fourier feature expansion
    - Gaussian basis expansion of distances
    - local-orbital basis
G025
  gloss: Problems and techniques around automating high-throughput computational workflows and ML pipelines, including model-selection bias, reproducibility, and reducing manual researcher intervention.
  terms:
    - SineCoulombMatrix
    - Visual warning system
    - CFID
    - Two-body descriptor
    - Three-body descriptor
    - Many-body descriptor
    - 2-body descriptor
    - 2b descriptor
    - SOAP descriptor with sparse points
    - SOAP sparse points
    - Light Gradient-Boosting Machine
    - Automated STM image analysis
    - STM image classification
    - Tersoff‑Hamann STM image simulation
    - sparse point selection
G026
  gloss: Methods and problems around charge equilibration, long-range electrostatics, and oxidation-state constraints in differentiable ML energy models.
  terms:
    - Franzblau algorithm
    - Fine‑tuning procedure using molecular vibrational spectra
    - Kernel normalization and exponentiation
    - Continuous expansion of JARVIS datasets and tools
    - Γ-phonon frequency prediction
    - Band structure and density‑of‑states calculation
    - Band gap prediction
    - Full phonon band structure prediction
    - Phonon spectra prediction
    - High entropy alloy phonon prediction
    - phonon physics
    - Phonon dispersion calculation
    - Bandgap calculation
    - Kernel normalisation and integer power
    - zero‑weight extra k‑path for band structures
G027
  gloss: Claims and datasets on DeepH, DeepH-E3, xDeepH, and HamGNN models predicting DFT Hamiltonians for twisted bilayer and magnetic 2D materials.
  terms:
    - Using the vdW‑DF‑OptB88 functional for all dimensionalities yields accurate lattice parameters and formation energies for both van‑der‑Waals and non‑vdW bonded materials.
    - DeepH accurately predicts DFT Hamiltonians and electronic properties for twisted bilayer bismuthene with strong spin‑orbit coupling, achieving errors comparable to those for twisted graphene.
    - DeepH‑E3 reproduces the band structure of magic‑angle twisted bilayer graphene (θ = 1.08°, 11 164 atoms) in excellent agreement with DFT and low‑energy continuum models, correctly capturing the flat bands near the Fermi level.
    - DeepH‑E3 generalizes across diverse material systems—including graphene, MoS₂, bismuthene, Bi₂Se₃, and Bi₂Te₃—achieving consistently sub‑meV Hamiltonian prediction errors.
    - MoS₂ 35×35 supercell
    - MoS₂ 35×35 supercell (efficiency test)
    - twisted bilayer bismuthene (TBB)
    - Construction of a Moiré‑twisted material database using DeepH‑E3
    - density functional theory
    - density functional perturbation theory
    - Accurate band‑structure prediction for twisted moiré materials
    - Modeling strong spin‑orbit coupling in twisted vdW materials
    - Study of twisted van der Waals heterostructures
    - Density of states calculation
    - Supercell approach
G028
  gloss: A mix of specific accuracy and speed claims for GAP, D4FT, and GNoME models, plus tasks on amorphous carbon and Si3N4 structure modeling.
  terms:
    - Tuning the exact‑exchange fraction α to 0.375 in HSE06 brings the calculated ZnO band gap into agreement with experiment.
    - GAP predicts graphite c parameter at 6.518 Å versus 6.625 Å from DFT, representing -1.6% overbinding.
    - FeCoNi alloys
    - High-temperature surface reconstructions of amorphous carbon
    - Limited system sizes in amorphous materials simulation
    - Large-scale amorphous carbon simulation
    - defect dynamics simulation
    - Liquid silicon structural analysis
    - Amorphous silicon generation and characterization
    - gap deformation potential screening
    - Gap deformation potential prediction
    - point defect analysis
    - Vacancy formation energy prediction
    - Nanoscale amorphous carbon structure simulations
    - liquid carbon simulation
G029
  gloss: Crystal/molecular graph neural network architectures (CGCNN, graph attention networks, MPNN) and claims about their scalability and performance.
  terms:
    - Interaction Network
    - Spectral Graph Convolution
    - Molecular Graph Convolutions (GC)
    - Kipf & Welling Graph Convolutional Networks
    - N-body Networks
    - Graph Neural Network Force Field
    - Graph convolution block
    - Graph Convolutional Network
    - graph neural network
    - Spectral graph convolution
    - graph neural networks with three-body interactions
    - Symmetry-adapted Graph Neural Network
    - crystal graph convolutional neural networks
    - Partial Node Assignment
    - Crystal Graph Attention Network
G030
  gloss: Techniques for uncertainty quantification (Gaussian process variance, Monte-Carlo dropout) and robust error/failure handling (auto-restart, outlier filtering) in ML pipelines.
  terms:
    - Bayesian uncertainty estimation
    - Error correction system
    - success rate
    - Regularized Gaussian‑process regression
    - Automatic error detection and correction
    - Unknown value imputation
G031
  gloss: Tasks and methods for predicting phonon spectra, IR/Raman activity, and vibrational properties of materials using machine learning models like Phonax and JARVIS.
  terms:
    - JARVIS-DFT contains approximately 40,000 materials and about 1 million calculated properties.
    - JARVIS‑FF includes data for roughly 1,500 materials and about 110 classical force‑fields.
    - JARVIS‑FF automatically computes bulk modulus, defect formation energies, and phonon spectra for each force‑field and compares them with DFT results to assess force‑field quality.
    - The derived ACE formulas enable efficient evaluation of forces and magnetic torques.
    - Higher degree representations (larger Lmax) better capture angular resolution and directional information critical for accurate force prediction and correlate with model expressivity.
    - MACE uses four-body (higher-order) messages, allowing only two message‑passing layers to achieve converged accuracy, whereas prior equivariant MPNNs required five or six layers.
    - Direct force prediction
    - TMetalFraction featurizer
    - Energy‑non‑conserving force predictions
    - angular resolution capture for force prediction
    - Force prediction accuracy
    - Interatomic force prediction
    - Force prediction
    - Force prediction for molecular dynamics
    - reverse communication of energy gradients
G032
  gloss: Components, techniques, and methods for building E(3)/SE(3)-equivariant graph neural networks, including tensor products, virtual nodes, and equivariant layers.
  terms:
    - PyTorch Geometric
    - Directional Message Passing Neural Network
    - Graph network models (CGCNN and MEGNet) have high errors on small datasets (<10^4 samples) but leverage large datasets more efficiently, outperforming traditional ML and AutoML approaches on tasks with approximately 10^4 data points or more.
    - Fully‑connected hidden layers
    - GPU‑accelerated tensor backend
    - GNN predictor
    - CGCNN convolution layers
    - STATIC
    - Static GNN
    - message-passing layers
    - convolutional LSTM
    - E(3)-equivariant neural network
    - Sparse Clebsch-Gordan exploitation
    - TensorBoard visualization
    - Message-passing process
G033
  gloss: Feature-encoding techniques using spherical harmonics and composition/atom-type encodings (one-hot, two-hot, composition-weighted) for equivariant network inputs.
  terms:
    - Spherical Channels
    - Spherical harmonics up to l=5
    - Spherical harmonic orientation encoding
    - QR‑based orthogonal constraint enforcement
    - Boundary‑condition encoding
    - Spherical harmonic projections
    - Atomic mass input features
    - Spherical harmonics Y
    - communication overhead in spatial decomposition
    - One-hot encoding
    - Multiple‑tower decomposition
    - Spherical harmonic encoding
    - QR decomposition
    - Categorical feature encoding
    - Wyckoff position analysis
G034
  gloss: Equivariant network designs (Tensor-Field Networks, SE(3)-Transformer, steerable CNNs) and techniques for building rotation-equivariant molecular/materials models.
  terms:
    - DeepH-E3 provides a universal E(3)-equivariant deep-learning framework that exactly preserves Euclidean symmetry of the DFT Hamiltonian, including spin-orbit coupling.
    - Filter-generating network
    - E(3)-equivariant convolution
    - Multiple MPNN models for orbital‑pair blocks
    - Single MPNN with multi‑dimensional output
    - End‑to‑end GNN predictor
    - PNA-based GNN predictor
    - Alternative GNN backbones
    - Extension of E(3)‑equivariant neural networks to capture DeepH’s small‑to‑large scale learning capability
    - deep tensor neural network
    - Transfer‑learning for small‑data properties
    - SiLU and Sigmoid gated activations
    - Gated multi‑layer perceptron
    - Spatiotemporal learning with ConvLSTM
    - Transfer learning for new loading conditions
G035
  gloss: Neural-network (Behler-Parrinello, Deep Potential) and empirical (Tersoff, Brenner, REBO) interatomic potentials for molecular dynamics, plus the problem of overfitting.
  terms:
    - ReaxFF
    - Neural Fingerprints
    - Tersoff potential
    - Deep neural network potential energy surface
    - Neural fingerprint
    - Martyna–Klein–Tuckerman barostat
    - Hillert model
G036
  gloss: Tasks and methods for predicting phonon spectra, IR/Raman activity, and vibrational properties of materials using machine learning models like Phonax and JARVIS.
  terms:
    - Franzblau algorithm
    - BANDS calculations use 128 k-points per segment for single element structures and 20 k-points for compounds along high-symmetry paths.
    - Computing Γ-phonon spectra for 146,323 Materials Project materials took less than five hours on an eight-GPU system, even for materials with over 400 atoms per unit cell.
    - The model was trained on a DFPT computational database containing phonon dispersion data for 1,521 crystalline inorganic materials.
    - Linear response phonon method
    - 2D heterostructure band‑alignment analysis
    - Phonon dispersion and vibrational thermodynamics
    - Phonon prediction in alloy systems
    - High entropy alloy phonon prediction
    - Electron-phonon coupling calculation
    - Electronic band structure calculation
    - Bandgap calculation
    - dynamic band‑count adjustment
    - zero‑weight extra k‑path for band structures
    - Direct phonon band prediction
G037
  gloss: A large, heterogeneous mix of claims, tasks, and methods spanning MLIP benchmarking, dataset properties, and large-scale atomistic simulation; diverse content, no single tight theme.
  terms:
    - Moment Tensor Potentials
    - ApproxNEB
    - Long-range correction schemes in machine learning interatomic potentials are essential for achieving state-of-the-art accuracy, improving in-distribution performance, and enabling significant gains in transferability to unseen regions of chemical space.
    - MLIP MD would unlock gains of orders of magnitude in accessible length and time scales versus AIMD, enabling study of new physical regimes.
    - MACE attains state‑of‑the‑art accuracy, achieving lower root‑mean‑square energy and force errors than previous models on the rMD17, 3BPA, and AcAc benchmark suites.
    - A single UMA model without any fine-tuning performs similarly or better than task-specialized models across diverse DFT tasks including materials, catalysis, molecules, molecular crystals, and MOFs.
    - Data efficiency in 3D molecular modeling
    - Efficient molecular dynamics simulation
    - Nucleation rate estimation
    - computational modeling
    - Atomic charge prediction
    - exchange-correlation functional selection
    - Composition-weighted mass encoding
    - Physically-motivated term addition
    - Multiple time-stepping algorithm
G038
  gloss: Mix of elastic-constant/modulus prediction tasks and DFT convergence parameters (k-point grids, energy/ionic thresholds); two loosely related themes.
  terms:
    - Approximately 150,000 compounds were found with energy distance less than 50 meV/atom from the convex hull.
    - Automatic k‑point convergence protocol
    - Dual grid technique
    - Plane-wave kinetic energy cut-off
    - 6×6 elastic constants matrix
    - Elasticity
    - Convex hull stability analysis
    - Sine Coulomb Matrix
    - lattice parameter
    - convex‑hull phase diagram construction
    - Automated k‑point and plane‑wave cutoff convergence
    - Weighted averaging of pure‑element volumes
    - Unit cell size filtering
    - High-symmetry path sampling
    - δEion convergence criterion
G039
  gloss: Wide-ranging claims and methods on machine-learned interatomic potentials (GAP, ACE, moment tensor potentials) and their accuracy relative to DFT for silicon and other systems.
  terms:
    - Behler‑Parrinello Neural Network
    - GW approximation
    - BANDS calculations use 128 k-points per segment for single element structures and 20 k-points for compounds along high-symmetry paths.
    - Regularized least‑squares solution for expansion coefficients
    - Extension of Gaussian Approximation Potentials to other materials
    - Development of transferable machine‑learning interatomic potentials with broader configurational coverage
    - Neural fingerprint
    - Vienna Ab Initio simulation Package
    - Tran-Blaha modified Becke-Johnson (TBmBJ) meta‑GGA
    - First-principles molecular dynamics
    - Noise reduction in atomistic data
    - Bridging computational and experimental data
    - Time-scale limitation in first-principles molecular dynamics
    - Phonon and vibrational property prediction
    - Band gap prediction
G040
  gloss: Methods and math behind the atomic cluster expansion (ACE), which generalizes MTP/SNAP/GAP potentials using spherical-harmonic expansions of tensorial atomic properties.
  terms:
    - Gaussian approximation potentials
    - Moment Tensor Potentials
    - ML potential extrapolation
    - generation of input files for cluster expansion
    - Hierarchical Body‑Order Expansion
    - Gaussian basis expansion of distances
G041
  gloss: Force, virial, and torque prediction methods and problems, including JARVIS-FF force-field benchmarking against DFT and global-state-conditioned loss functions.
  terms:
    - JARVIS‑FF includes data for roughly 1,500 materials and about 110 classical force‑fields.
    - Training SchNet with forces on only 1,000 examples yields better accuracy than training on 50,000 examples using energies alone, demonstrating that force information improves data efficiency.
    - Local atomic reference frame
    - MSE loss
    - Energy‑conserving force field learning
    - Force‑field quality assessment
    - Locality testing
    - Local reference frame construction
    - Energy‑force‑virial joint loss
    - Backpropagation for force and virial computation
G042
  gloss: Neural-network (Behler-Parrinello, Deep Potential) and empirical (Tersoff, Brenner, REBO) interatomic potentials for molecular dynamics, plus the problem of overfitting.
  terms:
    - Gaussian approximation potentials
    - Moment Tensor Potentials
    - ML potential extrapolation
    - generation of input files for cluster expansion
    - Hierarchical Body‑Order Expansion
    - Gaussian basis expansion of distances
G043
  gloss: A broad set of performance claims and proposed future extensions for ML property-prediction models (SchNet, MPNN, MEGNet, DeePMD, CGCNN), without one single topic.
  terms:
    - When applied to a set of 1,585 crystals with elastic properties, CGCNN predicts bulk modulus with a mean absolute error of 0.077 log(GPa) and shear modulus with a mean absolute error of 0.114 log(GPa), demonstrating good generalization.
    - Compared with a prior statistical‑learning approach, CGCNN achieves similar or lower root‑mean‑square errors for bulk modulus (0.075 log(GPa) vs 0.105) and shear modulus (0.127 log(GPa) vs 0.138) using the same training data.
    - When trained on MD17 trajectories with a combined energy‑and‑force loss, SchNet achieves energy MAE ≤0.12 kcal/mol and force MAE ≤0.33 kcal/mol/Å, matching or surpassing GDML and DTNN on all tested molecules, including flexible ones such as malonaldehyde and ethanol.
    - Including force information in the training loss causally improves SchNet’s generalization to previously unseen chemical structures in ISO17.
    - The ISO17 benchmark, comprising 645,000 conformations of 129 C7O2H10 isomers, is introduced to evaluate models on combined chemical and conformational variation.
    - On the QM9 benchmark, SchNet attains a mean absolute error of 0.31 kcal/mol for total‑energy prediction with 110 k training examples, outperforming DTNN (0.84) and enn‑s2s (0.45).
    - The DeePMD model provides excellent energy conservation in all tested systems up to 2 ns of NVE dynamics.
    - Simple MEGNet models that use only atomic number and spatial distance as input achieve chemical accuracy, perform comparably to full‑feature models, and outperform prior state‑of‑the‑art models on 8 of the 13 QM9 properties.
    - MEGNet models trained on approximately 60,000 crystals outperform prior models (SchNet and CGCNN) for formation energy, band gap, bulk modulus, and shear modulus, achieving errors comparable to or better than density‑functional‑theory uncertainties.
    - Ensembling two independently trained three‑block MEGNet models lowers the formation‑energy mean absolute error from 0.028 eV/atom to 0.024 eV/atom.
    - Graph networks constitute a universal machine‑learning framework that can accurately predict properties for both molecules and crystals.
    - DeePMD reproduces AIMD energies, forces, and virial tensors for water and ice with root‑mean‑square errors of approximately 1 meV per molecule for energy, 40 meV Å⁻¹ for forces, and 1–2 meV for the virial.
    - DeePMD accurately reproduces radial distribution functions and three‑body angular distribution functions of liquid water and ice, matching AIMD results.
    - Application of MEGNet models to a broader range of crystal properties beyond formation energy, band gap, and elastic moduli
    - Molecule property prediction
G044
  gloss: Mostly DeepH deep-learning DFT Hamiltonian prediction for twisted van der Waals materials, plus related equivariant (ACE) representations and GNN stress-prediction work in graphene.
  terms:
    - Inference with the MPNN takes roughly 10⁻² seconds per molecule, about 300,000 times faster than the DFT reference calculation (~10³ seconds).
    - GAP predictions for Young's modulus of amorphous carbon agree very well with experimental values at all relevant densities, while screened Tersoff potential captures the trend but significantly overestimates absolute values.
    - All fitting coefficients in the GAP model enter linearly, allowing solution via simple linear algebra rather than difficult nonlinear parameter optimization required for traditional potentials and neural networks.
    - Liquid carbon 5000K
    - DFT carbon reference data
    - Experimental sp3 data
    - ta-C surface energies
    - Accelerating meso‑level material design through AI‑driven defect‑property translation
    - Exploration of huge defect‑structure spaces
    - defect energetics and band‑gap underestimation
    - Extreme deformation potential detection
    - High‑level quantum‑chemical potential fitting
    - Interatomic potential fitting for silicon
    - Amorphous carbon surface (ta-C) modeling
    - liquid carbon simulation
G045
  gloss: Dominated by MD-17 and related small-molecule benchmark datasets (aspirin, ethanol, etc.) used to evaluate ML force fields like SchNet and NequIP.
  terms:
    - Interaction Network
    - Graph Convolutional Network
    - Kipf & Welling Graph Convolutional Networks
    - Duvenaud et al. Graph Convolutional Networks
    - Crystal Graph Convolutional Neural Networks
    - crystal graph networks
    - materials graph network
    - Multi-head attention
    - Extension of graph‑network models to additional state‑dependent material properties
    - Developing linear‑scaling IAPs based on graph deep learning for large‑scale molecular dynamics simulations of complex materials
    - Gated Graph Neural Networks
    - Materials Graph Network
    - graph neural networks with three-body interactions
    - Symmetry-adapted Graph Neural Network
    - Graph Networks
G046
  gloss: Claims and datasets on DeepH/xDeepH predicting DFT Hamiltonians for twisted van der Waals and 2D materials like graphene and CrI3, plus GNN mechanical property predictions.
  terms:
    - Silhouette score
    - MAE/ROC-AUC
    - ROC-AUC
    - F1‑score classification evaluation
    - ROC-AUC evaluation
    - silhouette score optimization
G047
  gloss: A mix of specific accuracy and speed claims for GAP, D4FT, and GNoME models, plus tasks on amorphous carbon and Si3N4 structure modeling.
  terms:
    - Symmetry‑adapted order‑parameter generation
    - Rotation transformation to global frame
    - Interstitial‑site topological search
    - Oriented FAST and Rotated BRIEF
    - atom-centered symmetry functions
    - Identification of interstitial sites
    - Symmetry compliance without architectural constraints
    - Learning rotational symmetry without hard constraints
    - Symmetry artifacts in automated workflows
    - rotational invariance enforcement
    - ISCD‑based elemental substitution
    - Chemical element permutation
    - Random rotation/inversion at each MD timestep
    - Jacobian symmetrization
    - statistical combinations of elemental properties
G048
  gloss: Energy-related prediction tasks (formation, exfoliation, activation energy) and their evaluation metrics (MAE, RMSE), across various ML models.
  terms:
    - DeepMD
    - Keras
    - Mode‑I and Mode‑II training pipelines
    - Cleaning stage
    - Training set cleaning pipeline
    - Iterative training workflow
    - Gradient-domain machine learning
    - Long-range behavior modeling
    - Heterogeneous training database fitting
    - Transfer‑learning for small‑data properties
    - Learning rate scheduling
    - Stochastic gradient descent
    - Softmax classification
    - Spatiotemporal learning with ConvLSTM
    - Deep transfer learning
G049
  gloss: Tasks, techniques, and components for predicting phonon spectra, density of states, and electronic band structure/band gaps.
  terms:
    - A general training method allows MPNNs to use larger node representation dimensions without increasing computation time or memory, providing substantial savings over previous high‑dimensional MPNNs.
    - Elemental embedding
    - EquiformerV2
    - CNN for NLP
    - Categorical feature encoding
    - Element embedding
G050
  gloss: A broad set of tasks and problems around predicting diverse material properties (structure, dielectric, glass formation) using JARVIS-ML/Matbench-style models.
  terms:
    - Machine‑learning stability prediction for crystal structures
    - Metal/semiconductor classification
    - Materials property prediction benchmarking
    - Crystal property prediction
    - Crystal structural relaxation
    - property prediction on MatBench benchmarks
    - Prediction of material properties
    - Materials property prediction via ML
    - Structure-based regression
    - Composition-only regression
    - machine learning for materials
    - Structural property prediction
    - Node‑wise regression
    - Rule‑of‑mixtures analysis
    - Nonlinear transformations
G051
  gloss: Message-passing GNN variants (directional, edge-update, vertex-update) and their component blocks for molecular/materials property prediction.
  terms:
    - Message Passing Neural Network (MPNN)
    - Davidson blocked scheme
    - Message passing networks
    - The Message Passing Neural Network framework unifies eight previously proposed graph‑based neural models under a single formalism.
    - Edge Network message function
    - Vertex and edge update blocks
    - Feed‑forward deep neural network
    - Message Passing layers
    - Node feature vectors
    - Edge feature vectors
    - Edge-conditioned neural network
    - Davidson blocked scheme
    - Edge‑dependent linear transformation
    - Virtual edges for long‑range communication
    - Directed message channels
G052
  gloss: A broad set of tasks, claims, and problems spanning materials discovery, property prediction, and high-throughput databases like the Materials Project and AFLOW.
  terms:
    - High‑throughput computational screening through the Materials Project has screened tens of thousands of compounds and identified promising candidates for solar water splitting, photovoltaics, topological insulators, scintillators, CO₂ capture, piezoelectrics, thermoelectrics, catalysis, hydrogen storage, and Li‑ion batteries, with experimental hits reported in several areas.
    - Matbench tasks include predicting optical, thermal, electronic, thermodynamic, tensile, and elastic properties given materials composition and/or crystal structure.
    - Foundation models enable cross-domain generalization and emergent capabilities for scientific discovery in materials science.
    - GNoME discovered over 2.2 million new stable inorganic materials and more than 45 000 new crystal prototypes, an order‑of‑magnitude increase over prior discovery efforts.
    - GNoME discovered 2.2 million crystal structures stable with respect to the Materials Project, with 381,000 newly on the convex hull, representing an order-of-magnitude expansion from all previous discoveries.
    - The AFLOW database currently contains approximately 400 experimental prototype structures and several million bcc/fcc/hcp‑derived superstructures.
    - SLME screening module
    - Efficient large‑scale simulation of heterovalent materials
    - Cross-material property comparison
    - Feature selection for materials property prediction
    - Materials design rule discovery
    - mechanical design
    - high-throughput materials discovery
    - Total energy prediction for crystals
    - Materials candidate screening
G053
  gloss: Techniques using Voronoi tessellation, QR, and CUR matrix decomposition for feature extraction and model parallelization, applied to copper/formate systems.
  terms:
    - Voronoi decomposition
    - The QR‑based reparameterization used in D4FT preserves the orthonormal constraints of Kohn‑Sham orbitals, making the search space equivalent to the original orthogonal function space, as proved in Proposition 3.1, and can be interpreted as projected gradient descent on the orthogonal constraint set.
    - Formate on Cu(110)
    - CUR matrix reconstruction
    - spatial decomposition
G054
  gloss: Model-fitting components (filter-generating and fitting networks) alongside data curation and filtering techniques, plus MSD-based free-energy analysis.
  terms:
    - GAP recovers three- and four-membered rings that are key features of liquid and low-density amorphous carbon structures, which empirical potentials (Tersoff/Brenner) fail to predict.
    - GAP predicts graphite c parameter at 6.518 Å versus 6.625 Å from DFT, representing -1.6% overbinding.
    - SiGe alloys
    - Disordered materials
    - Accurate, transferable interatomic potential for silicon
    - Computational cost of large-scale amorphous structure generation
    - Nonlocality in carbon allotropes
    - accurate interatomic potential for amorphous carbon
    - Gap-deformation potential prediction
    - point defect analysis
    - Defect energy prediction
    - SiO2 benchmark tests
    - amorphous materials
    - Atomistic simulations of liquid carbon
    - Interatomic potential development for carbon
G055
  gloss: A diverse mix of DFT software/codes (VASP, CASTEP, Quickstep), interatomic potentials, and materials datasets, especially solid electrolytes like LGPS, LLZO, and NASICON.
  terms:
    - ConvLSTM
    - EAM
    - Magpie
    - CGCNN
    - LOCO-CV
    - SNAP
    - CPMD
    - ConvLSTM network architecture
    - custodian
    - Li10GeP2S12
    - NASICON
    - Li10GeP2S12 (LGPS)
    - WannierTools
    - LAMMPS
    - Magpie featurizer
G056
  gloss: Optimizers (Adam, conjugate gradient), loss-function design, and tasks/problems around jointly fitting energies, forces, and stresses in interatomic models.
  terms:
    - Conjugate gradients
    - tight-binding
    - Machine‑learning models trained on Classical Force‑field Inspired Descriptors (CFID) achieve high predictive accuracy for formation energies, exfoliation energies, band gaps, magnetic moments, refractive indices, dielectric constants, thermoelectric performance, and piezoelectric/infrared modes.
    - C++ gradient module for descriptor derivatives
    - Adaptive loss‑prefactor schedule
    - GA optimizer
    - Nonlinear parameter optimization
    - Force locality quantification
    - Energy, force, and virial prediction
    - Atomic von Mises stress prediction
    - Force‑field benchmarking
    - machine learning benchmarking
    - Locality testing
    - Energy‑gradient force derivation
    - Auto‑differentiation for forces and stresses
G057
  gloss: Claims and components about CHGNet and M3GNet universal interatomic potentials, emphasizing charge- and magnetism-aware modeling and accuracy benchmarks.
  terms:
    - Coulomb Matrix
    - Representing tensors with spherical harmonics yields a dramatically sparser basis, where a rank‑N Cartesian tensor with 3^N components can be represented by (N+1)^2 spherical harmonics (e.g., a 10th‑order tensor requires 121 harmonics instead of 59 049 Cartesian components).
    - Gaussian‑basis distance expansion
    - Irreducible Basis Set B
    - Radial basis function edge attributes
    - Two-hot encoding for alloy elements
    - Gaussian RBF features at node level
    - Angular distribution function calculation
    - Spherical‑harmonic directional encoding
    - Radial basis functions
    - Spherical‑Harmonic Expansion of Cartesian Tensors
    - Radial basis function encoding
    - Spherical harmonics encoding
    - local-orbital basis
    - Spherical harmonics expansion
G058
  gloss: Cited works, methods, and components centered on message-passing neural network designs (directional, steerable, tensor field) for atomistic modeling.
  terms:
    - Message Passing Neural Network
    - PyTorch Geometric
    - Integrable deep neural networks
    - Message passing networks
    - Integrable Deep Neural Networks trained on derivative (chemical‑potential) data accurately learn the high‑dimensional free‑energy density and chemical‑potential functions for LiₓCoO₂, achieving mean‑squared errors comparable to the reported minima and reproducing DFT formation energies with a mean absolute error below 5 meV atom⁻¹.
    - The presented scale‑bridging framework, linking DFT‑informed statistical mechanics to continuum phase‑field models via IDNN‑learned free‑energy functions, constitutes the first systematic implementation of this kind for LiCoO₂ cathodes.
    - Weighted message‑passing convolution
    - Message Passing layers
    - Representation learning module
    - Tensor-Field Networks
    - Directional Message Passing
    - Spherical CNNs
    - Training of IDNN on chemical‑potential derivative data
    - Gradient‑based training of IDNN
    - Spatiotemporal learning with ConvLSTM
G059
  gloss: Tasks and results applying the GAP interatomic potential to amorphous and tetrahedral amorphous carbon structures, surfaces, and phase-diagram construction, benchmarked against DFT.
  terms:
    - For LGPS, the computed activation energy is 0.16 eV, which is slightly lower than experimental values (0.22 eV) but in line with previous computational studies (0.17-0.21 eV).
    - Heuristic + machine‑learning formation‑energy predictor
    - Fermi energy
    - Energy MAE
    - E RMSE
    - energy MAE (meV)
    - Gvrh MAE
    - MAE (eV/atom)
    - energy_RMS_eV
    - Formation energy regression
    - Energy prediction
    - Static energy calculation
    - Activation energy computation
    - Machine‑learning regression on formation energies
    - formation enthalpy calculation
G060
  gloss: Energy-related prediction tasks (formation, exfoliation, activation energy) and their evaluation metrics (MAE, RMSE), across various ML models.
  terms:
    - For LLZO, the computed activation energies range from 0.2 to 0.22 eV, in very good agreement with experimental values of 0.20-0.21 eV for cubic LLZO.
    - For LGPS, the computed activation energy is 0.16 eV, which is slightly lower than experimental values (0.22 eV) but in line with previous computational studies (0.17-0.21 eV).
    - Heuristic + machine‑learning formation‑energy predictor
    - Fermi energy
    - Materials Project formation energy
    - force MAE
    - Energy MAE
    - Gvrh MAE
    - MAE (eV/atom)
    - Formation energy prediction
    - Energy prediction
    - Static energy calculation
    - Activation energy computation
    - Machine‑learning regression on formation energies
    - MAE evaluation
G061
  gloss: A large, diverse mix of tasks, claims, and problems spanning materials property prediction, high-throughput discovery workflows, and specific case studies; no single tight theme.
  terms:
    - APL phonon library
    - Phonon DOS benchmark
    - SiO2 polymorph phonon calculations
    - band structure prediction
    - Electronic structure calculation
    - Phonon and elasticity calculations
    - phonon band structure calculation
    - Phonon dispersion and vibrational thermodynamics
    - hybrid functional band‑structure calculation
    - Band gap prediction
    - Materials property prediction from electronic bandstructure
    - Phonon dispersion calculation
    - Band structure calculation
    - Band structure calculations
    - Electronic structure determination
G062
  gloss: Mix of cited works, datasets, and methods spanning interatomic potentials, DFT codes, and materials databases; no single tight theme evident.
  terms:
    - EAM
    - GemNet
    - ANI-1
    - Quantum ESPRESSO
    - ICSD
    - RMM-DIIS
    - Altae-Tran et al.
    - OQMD database
    - MEGNET blocks
    - Kahle2020
    - Na3Zr2Si2PO12
    - i-PI
    - DimeNet
    - GemNet
    - PBE functional
G063
  gloss: Classical machine-learning methods (random forests, gradient boosting, clustering) and feature-importance/selection techniques, largely applied to property regression tasks like melting point prediction.
  terms:
    - Linear Atomic Cluster Expansion
    - Sparse cluster‑expansion surrogate
    - Decision‑tree based ML model
    - Permutation-invariant Aggregator
    - Cluster split method
    - Gaussian process regression
    - Random Forests
    - Atomic Cluster Expansion
    - cluster expansions
    - Materials clustering
    - Feature reduction
    - unsupervised clustering by bonding type
    - symbolic equation discovery
    - Operator-based feature derivation
    - Ensemble-averaged feature importance
G064
  gloss: A diverse mix of MD/DFT benchmark datasets, cited methods, and simulation components; no single unifying topic is evident.
  terms:
    - VASP
    - DTNN
    - AFLOW
    - Monkhorst-Pack
    - Hautier et al.
    - TIP3P
    - Benzene
    - Kahle2020
    - SSUB thermodynamic database
    - ta-C
    - MD-17 Azobenzene (original)
    - MD-17 Benzene (original)
    - Magpie featurizer
    - F1
    - BoltzTraP with constant relaxation time approximation
G065
  gloss: Tasks and results applying the GAP interatomic potential to amorphous and tetrahedral amorphous carbon structures, surfaces, and phase-diagram construction, benchmarked against DFT.
  terms:
    - NCV fold structure
    - Melt-quench trajectory generator
    - Amorphous carbon
    - Convex hull stability analysis
    - phase stability determination
    - Bulk crystal equation‑of‑state calculations
    - Phase diagram construction and stability assessment
    - Melt-quench simulation
    - Amorphous carbon surface (ta-C) modeling
    - liquid carbon simulation
    - surface reconstruction simulation
    - Bulk crystal structure modeling
    - Stability ranking
    - Melt-quench protocol
    - melt-quench trajectory generation
G066
  gloss: Tasks and methods for predicting phonon dispersion, IR/Raman activity, and electronic band gaps using models like Phonax trained on DFPT data.
  terms:
    - Phonax can predict molecular vibrational spectra and correctly assign IR/Raman active modes by analyzing irreducible representations.
    - A Γ-phonon database containing phonon spectra for 146,323 Materials Project materials was generated using the MVN approach, with unit cell atom counts ranging from 1 to 444 atoms.
    - Band gap
    - frozen phonon database
    - Linear response phonon method
    - Phonon prediction for complex materials
    - Phonon dispersion and density‑of‑states calculation
    - Γ-phonon spectra prediction
    - High entropy alloy phonon prediction
    - Raman spectrum calculation
    - Fine‑tuning with experimental vibrational spectra
    - Direct phonon band prediction
    - Band gap filtering
    - Alpha2F spectral function calculation
    - Band structure energy window selection
G067
  gloss: Materials discovery databases (Materials Project, OQMD) and high-throughput screening tasks including superconductor and general candidate discovery.
  terms:
    - pymatgen
    - By extracting per‑atom site energies in perovskites, the model reveals that large‑radius elements stabilize the A site and group 4‑6 elements stabilize the B site, and using these insights the high‑throughput search space for stable perovskites is reduced by a factor of seven (from 18,928 to 228 candidates).
    - The Materials Project provides an open RESTful API and a high‑level Python library (pymatgen) that enable programmatic querying and analysis of the full dataset.
    - AFLOW calculations follow a standardized sequential workflow: RELAX1 → RELAX2 → STATIC → BANDS for materials in the Elements, ICSD, and Heusler databases.
    - The PBE functional combined with PAW potentials is the default exchange-correlation functional and potential combination for ICSD, Binary Alloy, and Heusler databases.
    - Standardized Brillouin‑zone path generator
    - more challenging benchmark with chemical and structural variations
    - automate challenging chemical search problems in drug discovery
    - Exploiting intermediate PES data (energies, forces, stresses) from high‑throughput relaxations to further improve IAP accuracy
    - pymatgen structure matcher
    - energetic and structural properties
    - Accurate reproduction of AIMD structural properties
    - Lack of accessible high‑quality materials data
    - computational chemistry
    - Interactive materials design and rapid prototyping
G068
  gloss: Techniques and tasks for convex-hull stability classification and geometric/neighbor-based similarity analysis of atomic structures.
  terms:
    - Image‑based geometric matrix
    - Geometric matrix X
    - infinite neighbor graph
    - Convex hull stability analysis
    - Local geometry assessment
    - Geometry-barrier prediction independence
    - Local geometry comparison
    - Local geometry prediction
    - Confidence‑interval based neighbor selection
    - Neighbor sorting by species and inverse distance
    - Degree‑scaled aggregation
    - δ metric calculation
    - Convex hull analysis
    - Adaptive neighbor number
    - Kernel-based similarity measure
G069
  gloss: DFT+U variants (GGA+U, LDA+U, LSDA+Hubbard U) and problems of accuracy, speed, and reproducibility in high-throughput DFT versus ML/GDML alternatives.
  terms:
    - BAML
    - LDA+U
    - LDA
    - Liechtenstein DFT+U
    - AFLOW implements DFT+U using the Dudarev formalism with default Ueff values for a wide range of elements.
    - Input representation D_ij
    - Ueff parameters for d-block elements
    - U and J parameters for f-block elements
    - Benchmarking beyond‑DFT methods
    - Computational efficiency vs. performance trade-off
    - Computational cost in high-throughput DFT
    - System size limitations of DFT
    - High‑throughput DFT property calculation
    - GRU update
    - High‑throughput DFT calculations
G070
  gloss: Optimizers and scheduling techniques (Adam, learning-rate schedules, genetic algorithms) applied to structure relaxation and model training tasks.
  terms:
    - De‑noising non‑equilibrium structures (DeNS) auxiliary task
    - GA optimizer
    - Gradient predictions
    - Poor generalization to non-equilibrium configurations
    - Initial Structure to Relaxed Structure
    - DFT relaxation outcome prediction
    - Structure relaxations
    - Adam optimizer
    - Cosine learning rate scheduler
    - Cosine learning rate with linear warmup
    - LBFGS structural relaxation
    - TPOT genetic algorithm
    - Cyclical learning rate schedule
    - Two-step relaxation annealing
    - cosine learning rate schedule with linear warmup
G071
  gloss: Components and techniques for encoding interatomic distances and angles via radial (Gaussian, Bessel) and spherical-harmonic basis expansions.
  terms:
    - Coordinate‑free Wyckoff position encoding
    - Spherical‑harmonics embedding
    - Orthogonal Single‑Atom Basis Functions φ_v
    - Irreducible Basis Set B
    - Radial basis functions
    - Radial basis function edge attributes
    - Euclidean Rotary Positional Encoding
    - Hierarchical Body‑Order Expansion
    - Fourier angular basis
    - Spherical harmonic encoding
    - Inner product with spherical harmonics
    - Radial basis function encoding
    - Fourier space electrostatic calculation
    - Spherical harmonics tensor product
    - Tensor product with spherical harmonics
G072
  gloss: A list of specific named neural network architectures (GemNet, DimeNet, MEGNet, PhysNet, SevenNet, etc.) for interatomic potentials and property prediction.
  terms:
    - OrbNet
    - ResNet
    - NewtonNet
    - Architecture modifications reduced SevenNet-0 total model parameters from 16.24 million (GNoME) to 0.84 million.
    - MEGNET blocks
    - MEGNetLayer
    - MEGNet
    - DimeNet
    - ænet
