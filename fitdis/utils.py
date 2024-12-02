def load_pdf(lhaid="NNPDF40_nnlo_as_01180", member=0):
    import lhapdf

    # load the PDF set
    return lhapdf.mkPDF(lhaid, member)
