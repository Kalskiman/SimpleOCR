package fr.neatmonster.labs;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.Point;
import java.awt.RenderingHints;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import javax.swing.Box;
import javax.swing.JFrame;
import javax.swing.JPanel;

import fr.neatmonster.simpleann.Layer;
import fr.neatmonster.simpleann.Network;
import fr.neatmonster.simpleann.Neuron;
import fr.neatmonster.simpleann.data.Data;
import fr.neatmonster.simpleann.data.DataSet;
import fr.neatmonster.simpleann.functions.Sigmoid;
import fr.neatmonster.simpleann.neurons.BiasNeuron;
import fr.neatmonster.simpleann.neurons.InputNeuron;
import fr.neatmonster.simpleann.neurons.OutputNeuron;
import fr.neatmonster.simpleann.training.Backpropagation;

@SuppressWarnings("serial")
public class SimpleOCR extends JFrame implements Runnable {
    private class DrawingPanel extends JPanel {
        private BufferedImage image = null;
        private Graphics2D    g2d   = null;

        public DrawingPanel() {
            setPreferredSize(new Dimension(360, 360));
            final Point point = new Point();
            addMouseListener(new MouseAdapter() {

                @Override
                public void mousePressed(final MouseEvent e) {
                    point.setLocation(e.getPoint());
                }
            });
            addMouseMotionListener(new MouseAdapter() {

                @Override
                public void mouseDragged(final MouseEvent e) {
                    g2d.drawLine(point.x, point.y, e.getX(), e.getY());
                    point.setLocation(e.getPoint());
                    repaint();

                    right.setImage(image);
                }
            });
        }

        public void clear() {
            g2d.setColor(Color.WHITE);
            g2d.fillRect(0, 0, image.getWidth(), image.getHeight());
            g2d.setColor(Color.BLACK);
            repaint();

            right.setImage(image);
        }

        @Override
        public void paint(final Graphics g) {
            if (image == null) {
                image = new BufferedImage(360, 360, BufferedImage.TYPE_INT_RGB);
                g2d = (Graphics2D) image.getGraphics();
                g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                        RenderingHints.VALUE_ANTIALIAS_ON);
                g2d.setStroke(new BasicStroke(15f));
                clear();
            }
            g.drawImage(image, 0, 0, null);
        }
    }

    private class ImagePanel extends JPanel {
        private Image small;
        private Image giant;

        public ImagePanel() {
            setPreferredSize(new Dimension(360, 360));
        }

        public double[] getInput() {
            final BufferedImage bSmall = new BufferedImage(30, 30,
                    BufferedImage.TYPE_INT_RGB);
            bSmall.getGraphics().drawImage(small, 0, 0, null);
            final double[] input = new double[900];
            for (int y = 0; y < 30; ++y)
                for (int x = 0; x < 30; ++x)
                    input[x + y * 30] = bSmall.getRGB(x, y) < 0xffffffff ? 1f : 0f;
            return input;
        }

        @Override
        public void paint(final Graphics g) {
            if (giant == null) {
                g.setColor(Color.WHITE);
                g.fillRect(0, 0, 360, 360);
            } else
                g.drawImage(giant, 0, 0, null);
        }

        public void setImage(final BufferedImage image) {
            int minX = 0, maxX = 359, minY = 0, maxY = 359;
            searchX1: for (int x = 0; x < 360; ++x) {
                for (int y = 0; y < 360; ++y)
                    if (image.getRGB(x, y) < 0xffffffff)
                        break searchX1;
                minX = x;
            }
            searchX2: for (int x = 359; x >= 0; --x) {
                for (int y = 0; y < 360; ++y)
                    if (image.getRGB(x, y) < 0xffffffff)
                        break searchX2;
                maxX = x;
            }
            searchY1: for (int y = 0; y < 360; ++y) {
                for (int x = 0; x < 360; ++x)
                    if (image.getRGB(x, y) < 0xffffffff)
                        break searchY1;
                minY = y;
            }
            searchY2: for (int y = 359; y >= 0; --y) {
                for (int x = 0; x < 360; ++x)
                    if (image.getRGB(x, y) < 0xffffffff)
                        break searchY2;
                maxY = y;
            }
            if (minX == 359 || maxX == 0 || minY == 359 || maxY == 0)
                small = image.getScaledInstance(30, 30, Image.SCALE_FAST);
            else {
                final int dX = maxX - minX;
                final int dY = maxY - minY;
                final Image cut;
                if (dX > dY) {
                    int newY = minY - (dX - dY) / 2;
                    if (newY < 0)
                        newY = 0;
                    if (newY + dX > 360)
                        newY -= newY + dX - 360;
                    cut = image.getSubimage(minX, newY, dX, dX);
                } else {
                    int newX = minX - (dY - dX) / 2;
                    if (newX < 0)
                        newX = 0;
                    if (newX + dY > 360)
                        newX -= newX + dY - 360;
                    cut = image.getSubimage(newX, minY, dY, dY);
                }
                small = cut.getScaledInstance(30, 30, Image.SCALE_FAST);
            }
            giant = small.getScaledInstance(360, 360, Image.SCALE_FAST);
            repaint();

            if (training || drawing)
                return;

            final double[] input = getInput();
            final DataSet testSet = new DataSet();
            testSet.addData(input);
            network.evaluate(testSet);
            final double[] output = testSet.getData().get(0).getOutput();
            down.first = -1;
            for (int i = 0; i < 10; ++i)
                if (down.first < 0 || output[i] > output[down.first])
                    down.first = i;
            down.second = -1;
            for (int i = 0; i < 10; ++i)
                if (i != down.first
                        && (down.second < 0 || output[i] > output[down.second]))
                    down.second = i;
            double total = 0.0;
            for (final double prob : output)
                total += prob;
            down.firstProb = output[down.first] / total;
            down.secondProb = output[down.second] / total;
            down.repaint();
        }
    }

    private class InfoPanel extends JPanel {
        private int    first;
        private double firstProb;
        private int    second;
        private double secondProb;

        public InfoPanel() {
            setPreferredSize(new Dimension(730, 90));
        }

        @Override
        public void paint(final Graphics g) {
            String msg = "";
            String submsg = "";
            if (drawing) {
                msg = "Drawing mode";
                submsg = "{";
                for (int i = 0; i < 10; ++i)
                    submsg += i + ":" + trainSets[i].getSize() + ",";
                submsg = submsg.substring(0, submsg.length() - 1) + "}";
            } else if (training) {
                msg = "Training mode";
                submsg = "{";
                for (int i = 0; i < 10; ++i)
                    submsg += errRates[i] + ",";
                submsg = submsg.substring(0, submsg.length() - 1) + "}";
            } else {
                msg = "Estimate: " + first + " ("
                        + Math.round(firstProb * 1000.0) / 10.0 + "%)";
                submsg = "Second estimate: " + second + " ("
                        + Math.round(secondProb * 1000.0) / 10.0 + "%)";
            }

            final Graphics2D g2d = (Graphics2D) g;
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                    RenderingHints.VALUE_ANTIALIAS_ON);

            g2d.setColor(Color.BLACK);
            g2d.fillRect(0, 0, 730, 90);

            g2d.setColor(Color.WHITE);
            g2d.setFont(new Font("Rockwell", Font.PLAIN, 50));
            g2d.drawString(msg, 10, 45);
            g2d.setFont(new Font("Rockwell", Font.PLAIN, 25));
            g2d.drawString(submsg, 10, 75);
        }
    }

    public static void main(final String[] args) {
        new SimpleOCR().run();
    }

    private final DrawingPanel left  = new DrawingPanel();
    private final ImagePanel   right = new ImagePanel();
    private final InfoPanel    down  = new InfoPanel();
    private final Network      network;
    private final DataSet[]    trainSets;
    private final double[]     errRates;

    private boolean drawing  = false;
    private boolean training = false;

    public SimpleOCR() {
        setResizable(false);
        setTitle("NetOCR");
        setPreferredSize(new Dimension(730, 470));
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new FlowLayout(FlowLayout.LEADING, 0, 0));
        final JPanel upPanel = new JPanel();
        upPanel.setBackground(Color.BLACK);
        upPanel.setLayout(new FlowLayout(FlowLayout.LEFT, 0, 0));
        upPanel.add(left);
        upPanel.add(Box.createRigidArea(new Dimension(10, 360)));
        upPanel.add(right);
        add(upPanel);
        add(down);
        pack();
        setLocationRelativeTo(null);
        setVisible(true);

        addKeyListener(new KeyAdapter() {

            @Override
            public void keyPressed(final KeyEvent e) {
                switch (e.getKeyCode()) {
                case KeyEvent.VK_C:
                    left.clear();
                    break;
                case KeyEvent.VK_D:
                    drawing = !drawing;
                    training = false;
                    left.clear();
                    down.repaint();
                    break;
                case KeyEvent.VK_T:
                    drawing = false;
                    training = !training;
                    left.clear();
                    down.repaint();
                    break;
                case KeyEvent.VK_S:
                    ObjectOutputStream outStream = null;
                    try {
                        outStream = new ObjectOutputStream(
                                new FileOutputStream(new File("dataset")));
                        for (int i = 0; i < 10; ++i)
                            outStream.writeObject(trainSets[i]);
                    } catch (final Exception e1) {
                        e1.printStackTrace();
                    } finally {
                        try {
                            if (outStream != null)
                                outStream.close();
                        } catch (final Exception e2) {
                            e2.printStackTrace();
                        }
                    }
                    down.repaint();
                    break;
                case KeyEvent.VK_L:
                    ObjectInputStream inputStream = null;
                    try {
                        inputStream = new ObjectInputStream(
                                new FileInputStream(new File("dataset")));
                        for (int i = 0; i < 10; ++i) {
                            final DataSet trainSet = (DataSet) inputStream
                                    .readObject();
                            for (final Data data : trainSet.getData())
                                trainSets[i].addData(data);
                        }
                    } catch (final Exception e1) {
                        e1.printStackTrace();
                    } finally {
                        try {
                            if (inputStream != null)
                                inputStream.close();
                        } catch (final Exception e2) {
                            e2.printStackTrace();
                        }
                    }
                    down.repaint();
                    break;
                case KeyEvent.VK_0:
                case KeyEvent.VK_1:
                case KeyEvent.VK_2:
                case KeyEvent.VK_3:
                case KeyEvent.VK_4:
                case KeyEvent.VK_5:
                case KeyEvent.VK_6:
                case KeyEvent.VK_7:
                case KeyEvent.VK_8:
                case KeyEvent.VK_9:
                    if (drawing) {
                        final int d = e.getKeyCode() - KeyEvent.VK_0;
                        final double[] input = right.getInput();
                        final double[] output = new double[10];
                        output[d] = 1.0;
                        trainSets[d].addData(new Data(input, output));
                        left.clear();
                        down.repaint();
                    }
                    break;
                }
            }
        });

        network = new Network();
        network.addLayer(
                new Layer(900, InputNeuron.class).addNeuron(new BiasNeuron()));
        network.addLayer(new Layer(25, Neuron.class, new Sigmoid(1.0 / 25.0))
                .addNeuron(new BiasNeuron()));
        network.addLayer(new Layer(10, OutputNeuron.class, new Sigmoid(1.0)));
        Network.fullyConnect(network);

        trainSets = new DataSet[10];
        errRates = new double[10];
        for (int i = 0; i < 10; ++i) {
            trainSets[i] = new DataSet();
            errRates[i] = Double.NaN;
        }
    }

    @Override
    public void run() {
        while (true) {
            while (training) {
                for (int i = 0; i < 10; ++i) {
                    final Backpropagation training = new Backpropagation();
                    network.train(trainSets[i], training);
                    errRates[i] = training.getTotalError()
                            / trainSets[i].getSize() / Math.sqrt(10.0);
                    errRates[i] = Math.round(errRates[i] * 1000.0) / 10.0;
                }
                down.repaint();
            }
            try {
                Thread.sleep(100L);
            } catch (final InterruptedException e) {}
        }
    }
}
