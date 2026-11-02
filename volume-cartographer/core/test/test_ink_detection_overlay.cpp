#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <vc_object_store/objects/CatalogStore.hpp>
#include <doctest/doctest.h>

#include "CState.hpp"
#include "vc/ui/VCCollection.hpp"
#include "SurfaceCatalog.hpp"
#include "overlays/InkDetectionOverlayController.hpp"
#include "vc/core/types/Volume.hpp"
#include <nlohmann/json.hpp>
#include <vc_reader/zarr.hpp>
#include <vc_reader/zarr_write.hpp>
#include <filesystem>
#include <fstream>
#include "support/TestProject.hpp"
#include "vc/core/types/VolumePkg.hpp"
#include "vc/core/util/QuadSurface.hpp"

#include <QCoreApplication>
#include <QElapsedTimer>
#include <QImage>
#include <QThread>

#include <vc_object_store/objects/StoreReader.hpp>
#include <vc_object_store/objects/TransactionJson.hpp>
#include <vc_object_store/Uuid.hpp>

#include <opencv2/imgcodecs.hpp>

#include <memory>
#include <random>
#include <string>

namespace fs = std::filesystem;

namespace {

fs::path tempRoot(const std::string& tag)
{
    std::mt19937_64 rng(std::random_device{}());
    auto path = fs::temp_directory_path() /
                ("vc_ink_detection_overlay_" + tag + "_" + std::to_string(rng()));
    fs::create_directories(path);
    return path;
}

void writeFile(const fs::path& path, const std::string& body)
{
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << body;
    REQUIRE(out.good());
}

void writeGrayImage(const fs::path& path)
{
    fs::create_directories(path.parent_path());
    cv::Mat_<uint8_t> image(2, 3);
    image(0, 0) = 10;
    image(0, 1) = 20;
    image(0, 2) = 30;
    image(1, 0) = 40;
    image(1, 1) = 50;
    image(1, 2) = 60;
    REQUIRE(cv::imwrite(path.string(), image));
}

std::shared_ptr<QuadSurface> makeSurface(std::string id)
{
    cv::Mat_<cv::Vec3f> points(3, 3);
    points.setTo(cv::Vec3f(0.0f, 0.0f, 0.0f));
    auto surface = std::make_shared<QuadSurface>(points, cv::Vec2f(1.0f, 1.0f));
    surface->id = std::move(id);
    return surface;
}

// An Open Data sample store with one catalog surface published with an ink
// detection, listed by a catalog for the catalog volume.
struct CatalogFixture {
    fs::path root;
    CState state;
    std::shared_ptr<Volume> volume;
    std::unique_ptr<SurfaceCatalog> catalog;
    std::string surfaceUuid;
    fs::path inkFile;

    explicit CatalogFixture(const std::string& tag, const std::string& inkName)
        : root(tempRoot(tag))
    {
        namespace vo = vc::objects;
        const std::filesystem::path volumeRoot1 = root / "ct.zarr";
        const vcr::zarr::CreateOptions levelZero1{.shape = {4, 4, 4}, .chunks = {4, 4, 4}, .compressor = "", .overwrite = true};
        vcr::zarr::Writer::create(volumeRoot1 / "0", levelZero1);
        std::ofstream(volumeRoot1 / "meta.json") << nlohmann::json{{"type", "vol"}, {"uuid", "catalog-volume"}, {"name", "catalog-volume"}, {"format", "zarr"}, {"width", 4}, {"height", 4}, {"slices", 4}, {"voxelsize", 1.0}}.dump();
        volume = Volume::New(volumeRoot1);
        const auto storePath = root / "sample.vcobj";
        vo::OpenOptions storeOptions;
        storeOptions.author = "ink-test";
        {
            auto db = vo::ObjectStore::create(storePath, storeOptions);
            surfaceUuid = vo::util::newUuid();
            auto txn = db.write({{surfaceUuid, std::nullopt}});
            vo::ObjectRecord record;
            record.uuid = surfaceUuid;
            record.kind = "surface";
            record.name = "segment";
            record.meta = nlohmann::json{
                {"coordinates", {{"volume_uuid", "catalog-volume"}, {"scale_factor", 1}}},
                {"catalog", {{"sample", "sample"}, {"segment", "segment"}, {"files", {{
                    {"name", inkName},
                    {"label", "model-a"},
                    {"url", "https://example.test/" + inkName},
                }}}}}}.dump();
            db.upsertObject(txn, record);
            txn.commit();
        }
        state.objectStoreSession()->setGeometryFetcher("catalog",
            [](vo::ObjectStore&, const std::string&, const vo::ObjectStoreSession::FetchWrite&) {
                throw std::runtime_error("offline");
            });
        const auto attachment = state.objectStoreSession()->attach(storePath, storeOptions);
        // Published files are cached under VC3D's remote cache root: this test's.
        const auto config = root / "config";
        fs::create_directories(config);
        std::ofstream(config / "cache.json") << nlohmann::json{{"version", 1}, {"root", (root / "cache").string()}}.dump();
        qputenv("VC3D_CONFIG_DIR", QByteArray::fromStdString(config.string()));
        inkFile = vcr::zarr::decodedCachePath(vcr::zarr::selectedVc3dCacheRoot(), "https://example.test/" + inkName);
        catalog = std::make_unique<SurfaceCatalog>(state.objectStoreSession());
        catalog->setSource(attachment, volume, {volume->id()});
        QElapsedTimer timer;
        timer.start();
        while ((catalog->loading() || catalog->rows().empty()) && timer.elapsed() < 10000) {
            QCoreApplication::processEvents();
            QThread::msleep(1);
        }
        REQUIRE(catalog->rows().size() == 1);
    }
    std::shared_ptr<QuadSurface> surface() const
    {
        auto shown = makeSurface(surfaceUuid);
        shown->meta = nlohmann::json::parse(catalog->rows().begin()->second.record.meta);
        return shown;
    }
    ~CatalogFixture()
    {
        catalog.reset();
        std::error_code ec;
        fs::remove_all(root, ec);
    }
};

} // namespace

TEST_CASE("CState batches surface catalog mutations into one notification")
{
    int changedCount = 0;
    int willDeleteCount = 0;
    std::string changedName;
    CState state;
    auto first = makeSurface("first");
    auto second = makeSurface("second");

    const auto changedConnection = QObject::connect(
        &state, &CState::surfaceChanged,
        [&changedCount, &changedName](std::string name, std::shared_ptr<Surface>, bool) {
            ++changedCount;
            changedName = std::move(name);
        });
    const auto deleteConnection = QObject::connect(
        &state, &CState::surfaceWillBeDeleted,
        [&willDeleteCount](std::string, std::shared_ptr<Surface>) {
            ++willDeleteCount;
        });

    state.setSurfacesBatch({
        {"first", first},
        {"second", second},
    });

    CHECK(changedCount == 1);
    CHECK(changedName.empty());
    CHECK(willDeleteCount == 0);
    CHECK(state.surface("first") == first);
    CHECK(state.surface("second") == second);

    state.setSurfacesBatch({
        {"first", nullptr},
        {"second", first},
    });

    CHECK(changedCount == 2);
    CHECK(changedName.empty());
    CHECK(willDeleteCount == 0);
    CHECK(state.surface("first") == nullptr);
    CHECK(state.surface("second") == first);

    QObject::disconnect(changedConnection);
    QObject::disconnect(deleteConnection);
}

TEST_CASE("CState::closeAll clears the project's surfaces and keeps the view planes")
{
    for (const bool withProject : {false, true}) {
        CState state;
        if (withProject) {
            const auto store = vc::test::createTestProject();
            state.setVpkg(VolumePkg::open(store.path));
        }
        state.setSurface("segmentation", makeSurface("segmentation"));
        state.setSurface("other", makeSurface("other"));
        const auto xy = state.surface("xy plane");
        REQUIRE(xy);

        state.closeAll();

        CHECK(state.surface("segmentation") == nullptr);
        CHECK(state.surface("other") == nullptr);
        CHECK(state.surface("xy plane") == xy);
        CHECK(state.surface("xz plane"));
        CHECK(state.surface("yz plane"));
    }
}

TEST_CASE("ink detection options follow the active catalog surface")
{
    int argc = 0;
    QCoreApplication app(argc, nullptr);
    CatalogFixture fixture("options", "prediction.jpg");
    writeFile(fixture.inkFile, "not-empty");

    auto source = fixture.surface();
    fixture.state.setSurface("segmentation", source, false, false);

    InkDetectionOverlayController overlay(&fixture.state);
    overlay.setSurfaceCatalog(fixture.catalog.get());
    REQUIRE(overlay.options().size() == 1);
    CHECK(overlay.options().front().localPath == fixture.inkFile);
    CHECK(overlay.options().front().name == "prediction.jpg");

    // A preview of the same surface keeps them; another surface has none.
    fixture.state.setSurface("segmentation", fixture.surface(), false, false);
    CHECK(overlay.options().size() == 1);
    fixture.state.setSurface("segmentation", makeSurface({}), false, false);
    CHECK(overlay.options().empty());
}

TEST_CASE("ink detection overlay flips the displayed image")
{
    int argc = 0;
    QCoreApplication app(argc, nullptr);
    CatalogFixture fixture("flip", "prediction.png");
    writeGrayImage(fixture.inkFile);

    fixture.state.setSurface("segmentation", fixture.surface(), false, false);

    InkDetectionOverlayController overlay(&fixture.state);
    overlay.setSurfaceCatalog(fixture.catalog.get());
    REQUIRE(overlay.options().size() == 1);
    overlay.setSelectedPath(overlay.options().front().localPath);

    REQUIRE(!overlay.selectedImageForTesting().isNull());
    CHECK(qRed(overlay.selectedImageForTesting().pixel(0, 0)) == 10);
    CHECK(qAlpha(overlay.selectedImageForTesting().pixel(0, 0)) == 10);

    overlay.setHorizontalFlip(true);
    CHECK(overlay.horizontalFlip());
    CHECK(qRed(overlay.selectedImageForTesting().pixel(0, 0)) == 30);
    CHECK(qRed(overlay.selectedImageForTesting().pixel(2, 0)) == 10);

    overlay.setVerticalFlip(true);
    CHECK(overlay.verticalFlip());
    CHECK(qRed(overlay.selectedImageForTesting().pixel(0, 0)) == 60);
    CHECK(qRed(overlay.selectedImageForTesting().pixel(2, 1)) == 10);

    overlay.setHorizontalFlip(false);
    CHECK(!overlay.horizontalFlip());
    CHECK(qRed(overlay.selectedImageForTesting().pixel(0, 0)) == 40);
    CHECK(qRed(overlay.selectedImageForTesting().pixel(2, 1)) == 30);
}

TEST_CASE("ink detection options include the surface's ink channels")
{
    int argc = 0;
    QCoreApplication app(argc, nullptr);
    CState state;
    auto surface = makeSurface("segment");
    // Ink at two pixels per grid step, as `vc_object_ingest --surface-channel` stores it.
    cv::Mat_<uint8_t> ink(6, 6, uint8_t(0));
    ink(0, 0) = 200;
    ink(5, 5) = 50;
    surface->setChannel("ink_model_b", ink);
    surface->setChannel("ink", cv::Mat_<uint8_t>(3, 3, uint8_t(9)));
    surface->setChannel("generations", cv::Mat_<uint16_t>(3, 3, uint16_t(7)));
    surface->setChannel("inked", cv::Mat_<uint8_t>(3, 3, uint8_t(1)));
    state.setSurface("segmentation", surface, false, false);

    InkDetectionOverlayController overlay(&state);
    overlay.refreshAvailableDetections();
    REQUIRE(overlay.options().size() == 2);
    CHECK(overlay.options()[0].channel == "ink");
    CHECK(overlay.options()[1].channel == "ink_model_b");
    CHECK(overlay.options()[1].label == "ink_model_b");
    CHECK(!fs::exists(overlay.options()[1].localPath));

    overlay.setSelectedPath(overlay.options()[1].localPath);
    REQUIRE(overlay.hasLoadedSelection());
    CHECK(overlay.selectedIsSingleChannel());
    const QImage& image = overlay.selectedImageForTesting();
    CHECK(image.width() == 6);
    CHECK(image.height() == 6);
    CHECK(qRed(image.pixel(0, 0)) == 200);
    CHECK(qRed(image.pixel(5, 5)) == 50);
    CHECK(qAlpha(image.pixel(1, 1)) == 0);

    // Another surface has none of them.
    state.setSurface("segmentation", makeSurface("other"), false, false);
    CHECK(overlay.options().empty());
    CHECK(!overlay.hasLoadedSelection());
}

TEST_CASE("ink detection overlay reads an ink channel from the object store")
{
    namespace vo = vc::objects;
    int argc = 0;
    QCoreApplication app(argc, nullptr);
    const auto root = tempRoot("store_channel");
    auto session = std::make_shared<vo::ObjectStoreSession>();
    vo::OpenOptions options;
    options.author = "ink-test";
    { auto created = vo::ObjectStore::create(root / "objects.vcobj", options); }
    const auto attachment = session->attach(root / "objects.vcobj", options);

    cv::Mat_<cv::Vec3f> points(3, 3);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) points(r, c) = {float(c), float(r), 10.f};
    QuadSurface source(points, {1.0f, 1.0f});
    cv::Mat_<uint8_t> ink(6, 6);
    for (int r = 0; r < 6; ++r)
        for (int c = 0; c < 6; ++c) ink(r, c) = uint8_t(10 * r + c);
    source.setChannel("ink", ink);
    const auto uuid = vo::util::newUuid();
    {
        vo::ObjectRecord record;
        record.uuid = uuid;
        record.kind = "surface";
        record.name = "segment";
        auto request = session->newRequest(attachment);
        request.expectedObjects = {{uuid, std::nullopt}};
        request.operations.push_back(vo::CreateObjectOp{vo::objectRecordJson(record)});
        request.operations.push_back(vo::PutGeometryOp{uuid, quad_surface_geometry(source), {}});
        session->submit(attachment, request);
    }

    std::shared_ptr<QuadSurface> loaded;
    std::string revision;
    session->readStore(attachment, [&](const vo::StoreReader& reader, const auto&) {
        loaded = load_quad_from_store(reader, uuid);
        revision = reader.currentGeometry(uuid)->uuid;
    });
    loaded->setChannelLoader(store_channel_loader(session, attachment, revision));
    REQUIRE(!loaded->channelLoaded("ink"));

    CState state;
    state.setSurface("segmentation", loaded, false, false);
    InkDetectionOverlayController overlay(&state);
    overlay.refreshAvailableDetections();
    REQUIRE(overlay.options().size() == 1);
    overlay.setSelectedPath(overlay.options().front().localPath);
    REQUIRE(overlay.hasLoadedSelection());
    const QImage& image = overlay.selectedImageForTesting();
    CHECK(image.width() == 6);
    CHECK(image.height() == 6);
    CHECK(qRed(image.pixel(3, 2)) == 23);
    CHECK(qRed(image.pixel(5, 5)) == 55);

    state.setSurface("segmentation", nullptr, false, false);
    loaded.reset();
    session.reset();
    std::error_code ec;
    fs::remove_all(root, ec);
}

TEST_CASE("Project stores are peers with independent persistent per-kind selections")
{
    const auto project = vc::test::createTestProject();
    const auto other = vc::test::createTestProject();
    CState state;
    state.setVpkg(project.open());
    const auto first = state.storeFor("surface");
    REQUIRE(first);
    const auto second = state.attachStore(other.storePath, "vc-test");
    REQUIRE(state.attachedStores().size() == 2);
    CHECK(state.storeFor("fiber") == first);
    for (const auto& kind : CState::routableKinds()) {
        state.setStoreFor(kind, second);
        CHECK(state.storeFor(kind) == second);
        state.setStoreFor(kind, first);
    }
    state.setStoreFor("fiber", second);
    state.setStoreFor("umbilicus", second);
    CHECK(state.storeFor("surface") == first);
    CHECK(state.storeFor("volume_transform") == first);
    auto document = vc::project::ProjectDocument::load(project.path);
    CHECK(document.data()["selections"]["fiber"] != document.data()["selections"]["surface"]);
    state.prepareStoreSwitch = [] { return false; };
    CHECK_THROWS(state.setStoreFor("fiber", first));
    CHECK(state.storeFor("fiber") == second);
    state.prepareStoreSwitch = {};
    state.detachStore(second);
    CHECK_FALSE(state.storeFor("fiber"));
    CHECK_FALSE(state.storeFor("umbilicus"));
    CHECK(fs::exists(other.storePath));
    CHECK_THROWS(state.detachStore(*first));
    state.closeProject();
    state.setVpkg(project.open());
    CHECK_FALSE(state.storeFor("fiber"));
    CHECK(state.storeFor("surface"));
}

TEST_CASE("Unavailable store references survive opening and can be relocated")
{
    const auto project = vc::test::createTestProject();
    const auto other = vc::test::createTestProject();
    auto document = vc::project::ProjectDocument::load(project.path);
    auto data = document.data();
    data["stores"].push_back({{"id", "unavailable"}, {"location", "missing.vcobj"}, {"read_only", true}});
    data["selections"]["fiber"] = "unavailable";
    document.replace(data);
    CState state;
    state.setVpkg(project.open());
    CHECK(state.hasVpkg());
    CHECK_FALSE(state.storeFor("fiber"));
    CHECK(state.storeFor("surface"));
    CHECK(state.vpkg()->document().data() == data);
    state.replaceStoreReference("unavailable", other.storePath);
    REQUIRE(state.storeFor("fiber"));
    CHECK_FALSE(state.objectStoreSession()->writable(*state.storeFor("fiber")));
    CHECK(state.vpkg()->document().data()["selections"]["fiber"] == "unavailable");
}

TEST_CASE("A failed point-collection save prevents switching its store")
{
    const auto project = vc::test::createTestProject();
    const auto other = vc::test::createTestProject();
    CState state;
    state.setVpkg(project.open());
    const auto first = state.storeFor("point_collection");
    REQUIRE(first);
    const auto second = state.attachStore(other.storePath, "vc-test");
    const std::filesystem::path volumeRoot2 = vc::test::testProjectDirectory() / "point-save-failure";
    const vcr::zarr::CreateOptions levelZero2{.shape = {4, 4, 4}, .chunks = {4, 4, 4}, .compressor = "", .overwrite = true};
    vcr::zarr::Writer::create(volumeRoot2 / "0", levelZero2);
    std::ofstream(volumeRoot2 / "meta.json") << nlohmann::json{{"type", "vol"}, {"uuid", "point-save-failure"}, {"name", "point-save-failure"}, {"format", "zarr"}, {"width", 4}, {"height", 4}, {"slices", 4}, {"voxelsize", 1.0}}.dump();
    auto volume = Volume::New(volumeRoot2);
    state.setCurrentVolume(volume);
    state.pointCollection()->addPoint("unsaved", {1, 2, 3});
    const auto before = state.vpkg()->document().data();
    // Simulate a lost store connection while the debounce has pending work.
    state.objectStoreSession()->detach(*first);
    CHECK_THROWS(state.setStoreFor("point_collection", second));
    CHECK(state.storeFor("point_collection") == first);
    CHECK(state.vpkg()->document().data() == before);
    CHECK(state.pointCollection()->getCollectionId("unsaved").has_value());
    CHECK_THROWS(state.savePointCollections()); // Still pending, not silently dropped.
    CHECK_THROWS(state.closeAll());
    CHECK(state.pointCollection()->getCollectionId("unsaved").has_value());
}
