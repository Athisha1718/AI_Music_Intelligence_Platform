from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

styles = getSampleStyleSheet()

def create_report(filename,
                  genre,
                  confidence,
                  music_dna,
                  recommendations,
                  output_file):

    pdf = SimpleDocTemplate(output_file)

    story = []

    story.append(
        Paragraph("<b>AI Music Intelligence Platform</b>", styles["Title"])
    )

    story.append(
        Paragraph("<br/>", styles["BodyText"])
    )

    story.append(
        Paragraph(f"<b>Audio File:</b> {filename}", styles["BodyText"])
    )

    story.append(
        Paragraph(f"<b>Predicted Genre:</b> {genre}", styles["BodyText"])
    )

    story.append(
        Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles["BodyText"])
    )

    story.append(
        Paragraph("<br/><b>Music DNA</b>", styles["Heading2"])
    )

    for key, value in music_dna.items():

        story.append(
            Paragraph(f"{key} : {value}", styles["BodyText"])
        )

    story.append(
        Paragraph("<br/><b>Recommended Songs</b>", styles["Heading2"])
    )

    for song in recommendations:

        story.append(
            Paragraph(
                f"{song['filename']} ({song['genre']}) - {song['score']}%",
                styles["BodyText"]
            )
        )

    pdf.build(story)