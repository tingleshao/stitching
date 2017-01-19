import cv2
import imutils
import numpy as np
import argparse


#TODO: try using a bundle adjustment method and see the differences



def stitch_all_images_bundle(images, width, height, ratio=0.75, reprojThresh=4.0, showMatches=False):
    result = None
    return result 


def stitch_all_images(images, width, height, ratio=0.75, reprojThresh=4.0, showMatches=False):
    # width; number of microcameras per row
    # height: number of microcameras per column
    print "stitching all..."
    num_of_images = len(images)
    rows = []
    for j in xrange(height):
        row_images = images[j*width:(j+1)*width]
        result = None
        for i in xrange(width-1):
            imageB = row_images[i+1]
            if i == 0:
                imageA = row_images[i] # todo:later change it to existing stitched images
            else:
                imageA = result
            (kpsA, featuresA) = detect_and_describe(imageA)
            (kpsB, featuresB) = detect_and_describe(imageB)
            M = match_key_points(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
            if M is None:
                return None
            (matches, H, status) = M
            new_result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
            new_result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
            if showMatches:
                vis = draw_matches(imageA, imageB, kpsA, kpsB, matches, status)
            result = new_result
        rows.append(result)
    print "len rowsA; " + str(len(rows))
    result = None
    for j in xrange(height-1):
        imageB = rows[j+1]
        imageA = rows[j]
        (kpsA, featuresA) = detect_and_describe(imageA)
        (kpsB, featuresB) = detect_and_describe(imageB)
        M = match_key_points(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
        if M is None:
            return None
        (matches, H, status ) = M
        print " image A shape 1: " + str(imageA.shape[1])
        print " image A shape 0 + image B shape 0: "  + str(imageA.shape[0]) + "  " + str(imageB.shape[0])
        print " image B shape 1: " + str(imageB.shape[1])
        new_result = cv2.warpPerspective(imageA, H, (imageA.shape[1], imageA.shape[0]+imageB.shape[0]))
        new_result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        if showMatches:
            vis = draw_matches(imageA, imageB, kpsA, kpsB, matches, status)
        result = new_result
    return result


def stitch_images(images, ratio=0.75, reprojThresh=4.0, showMatches=False):
    print "stitching..."
    (imageB, imageA) = images
    (kpsA, featuresA) = detect_and_describe(imageA)
    (kpsB, featuresB) = detect_and_describe(imageB)
    M = match_key_points(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
    if M is None:
        return None
    (matches, H, status) = M
    result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    print str(imageA.shape)
    print str(imageB.shape)
    print "shape: " + str(result.shape)
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    if showMatches:
        vis = draw_matches(imageA, imageB, kpsA, kpsB, matches, status)
        return (result, vis)
    return result


def detect_and_describe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)

    kps = np.float32([kp.pt for kp in kps])
    return (kps, features)


def match_key_points(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    if len(matches) > 4:
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        return (matches, H, status)
    return None


def draw_matches(imageA, imageB, kpsA, kpsB, matches, status):
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA+wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        if s == 1:
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="""mode""", type=int)
    args = parser.parse_args()
    test_single_pair = False
    test_every_pair = False
    if args.m == 0:
        test_single_pair = True
    elif args.m == 1:
        test_every_pair = True
    if test_single_pair:
        imageA = cv2.imread("images/first.jpg")
        imageB = cv2.imread("images/second.jpg")
        imageA = imutils.resize(imageA, width=400)
        imageB = imutils.resize(imageB, width=400)
        (result, vis) = stitch_images([imageA, imageB], showMatches=True)
        cv2.imshow("Image A", imageA)
        cv2.imshow("Image B", imageB)
        cv2.imshow("Keypoint Matches", vis)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
    else:
        images = []
        for i in xrange(18):
            file_name = "images/mcam_" + str(i+1) + "_scale_2.jpg"
            curr_image = cv2.imread(file_name)
            curr_image = imutils.resize(curr_image, width=400)
            images.append(curr_image)
        result = stitch_all_images(images, 6, 3, showMatches=False)
    #    for i in xrange(len(results)):
    #        print i
        cv2.imshow("Result" + str(i), result)
        cv2.waitKey(0)


if __name__  == "__main__":
    main()
