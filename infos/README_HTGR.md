# HTGR Dust Management Documentation

This directory contains LaTeX documentation on radioactive dust management in pebble-bed High-Temperature Gas-Cooled Reactors (HTGRs).

## Files

- `htgr_dust_management.tex` - Main LaTeX document containing a comprehensive subsection on proposed solutions for radioactive dust removal and management
- `references.bib` - Bibliography file with placeholder citations

## Content Overview

The document discusses:

1. **Primary Loop and Outlet Filtration Systems**
   - Electrostatic precipitators
   - Scrubber systems
   - HEPA filters and high-temperature filter media

2. **Helium Purification and Sampling Systems**
   - Auxiliary purification loops
   - Centrifugal separators and cyclonic devices
   - Multi-stage filtration assemblies

3. **Maintenance and Shutdown-Based Cleaning Protocols**
   - Vacuum extraction
   - Surface washing and mechanical cleaning
   - Remote handling equipment considerations

4. **Operational Constraints and Design Considerations**
   - Online refueling capability impacts
   - Bottom-outlet configuration challenges
   - Sub-micron particle capture difficulties (~1 µm and below)

## Compilation

To compile the document (requires a LaTeX distribution):

```bash
cd infos
pdflatex htgr_dust_management.tex
bibtex htgr_dust_management
pdflatex htgr_dust_management.tex
pdflatex htgr_dust_management.tex
```

## Citations

The document includes placeholder citations:
- `Moormann2018Caution` - HTR dust explosion characteristics
- `Xie2017HTR10Dust` - HTR-10 dust characterization studies

These are placeholder references and should be replaced with actual bibliographic data when available.
