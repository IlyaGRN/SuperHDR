#ifndef SUPERHDR_H
#define SUPERHDR_H

#include <QtWidgets/QDialog>
#include <qgraphicsscene.h>
#include "ui_superhdr.h"
#include "SuperHDRProcessor.h"
#include <opencv2\core\core.hpp>

class SuperHDR : public QDialog
{
	Q_OBJECT

public:
	SuperHDR(QWidget *parent = 0);
	~SuperHDR();

private:
	SuperHDRProcessor *shdr_processor;
	Ui::SuperHDRClass ui;

	bool m_loaded_img;
	bool m_HDR;

	QString *fileName;

	cv::Mat *superHdrOutput;
	QImage *input_image;
	QImage *hdr_image;

	QGraphicsPixmapItem* image_item;
	QGraphicsScene *main_graphic_scene;

	void SetImageView(const QImage &image);

	private slots:

	void on_pushButtonLoad_clicked();
	void on_pushButtonRunHdr_clicked();
	void on_pushButtonSave_clicked();
	void on_radioButtonOriginal_clicked();
	void on_radioButtonSuperhdr_clicked();

};

#endif // SUPERHDR_H
