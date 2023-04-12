#include "superhdr.h"
#include <QFileDialog.h>
#include <QMessageBox.h>
#include <QGraphicsPixmapItem>

SuperHDR::SuperHDR(QWidget *parent)
	: QDialog(parent)
{
	ui.setupUi(this);

	m_loaded_img = false;
	m_HDR = false;
	main_graphic_scene = NULL;
	input_image = NULL;
	hdr_image = NULL;
	image_item = NULL;
	fileName = NULL;

	shdr_processor = SuperHDRProcessor::GetInstance();




	ui.pushButtonRunHdr->setEnabled(false);
	ui.pushButtonSave->setEnabled(false);
	ui.radioButtonOriginal->hide();
	ui.radioButtonSuperhdr->hide();

}

SuperHDR::~SuperHDR()
{
	if (main_graphic_scene != NULL) {
		delete main_graphic_scene;
	}
	if (input_image != NULL) {
		delete input_image;
	}
	if (fileName != NULL) {
		delete fileName;
	}
}


void SuperHDR::on_pushButtonLoad_clicked() {

	if (main_graphic_scene == NULL) {
		main_graphic_scene = new QGraphicsScene();
	}
	else {
		delete main_graphic_scene;
		main_graphic_scene = new QGraphicsScene();
	}

	fileName = new QString(QFileDialog::getOpenFileName(this,
		tr("Open Address Book"), "",
		tr("JPEG (*.jpg);;PNG (*.png);;TIF (*.tif);;All Files (*)")));

	if (*fileName != "") {

		if (input_image != NULL) {
			delete input_image;
			input_image = NULL;
		}
		if (hdr_image != NULL) {
			delete hdr_image;
			hdr_image = NULL;
		}

		input_image = new QImage(*fileName);

		if (input_image->isNull()) {
			QMessageBox::information(this, "Image Viewer", "Error Displaying image");
			return;
		}

		/*if (image_item != NULL) {
		delete image_item;
		}*/

		SetImageView(*input_image);

		m_HDR = false;
		m_loaded_img = true;
		ui.radioButtonOriginal->show();
		ui.radioButtonSuperhdr->show();
		ui.radioButtonOriginal->setChecked(true);
		ui.radioButtonSuperhdr->setEnabled(false);
		ui.pushButtonRunHdr->setEnabled(true);

	}

	else {
		m_loaded_img = false;
	}
}

void SuperHDR::on_pushButtonRunHdr_clicked() {

	if (!m_HDR && m_loaded_img) {

		if (hdr_image != NULL) {
			delete hdr_image;
			hdr_image = NULL;
		}

		shdr_processor->init(fileName->toStdString());
		shdr_processor->run();
		superHdrOutput = shdr_processor->GetOutput();

		hdr_image = new QImage((uchar*)superHdrOutput->data, superHdrOutput->cols, superHdrOutput->rows, superHdrOutput->step, QImage::Format_RGB888);
		SetImageView(*hdr_image);
		m_HDR = true;

		ui.radioButtonOriginal->setChecked(false);
		ui.radioButtonSuperhdr->setChecked(true);
		ui.radioButtonSuperhdr->setEnabled(true);
		ui.pushButtonSave->setEnabled(true);

	}
}


void SuperHDR::on_pushButtonSave_clicked() {

	QString save_file = QString(QFileDialog::getSaveFileName(this,
		tr("Open Address Book"), "",
		tr("JPEG (*.jpg);;PNG (*.png);;TIF (*.tif);;All Files (*)")));

	if (save_file != "") {
		cv::cvtColor(*superHdrOutput, *superHdrOutput, CV_RGB2BGR);
		imwrite(save_file.toStdString(), *superHdrOutput);
	}
}

void SuperHDR::SetImageView(const QImage &image) {

	main_graphic_scene->setSceneRect(0, 0, image.width(), image.height());
	/*if (image_item != NULL) {
	main_graphic_scene->removeItem(image_item);
	}*/

	if (image.height()>ui.graphicsView->height() && image.width()>ui.graphicsView->width())
	{
		ui.graphicsView->ensureVisible(main_graphic_scene->sceneRect());
		ui.graphicsView->fitInView(main_graphic_scene->sceneRect(), Qt::KeepAspectRatio);
		ui.graphicsView->fitInView(main_graphic_scene->itemsBoundingRect(), Qt::KeepAspectRatio);
	}
	image_item = new QGraphicsPixmapItem(QPixmap::fromImage(image));
	main_graphic_scene->addItem(image_item);
	ui.graphicsView->setScene(main_graphic_scene);
}

void SuperHDR::on_radioButtonOriginal_clicked() {
	SetImageView(*input_image);
}

void SuperHDR::on_radioButtonSuperhdr_clicked() {
	SetImageView(*hdr_image);
}

