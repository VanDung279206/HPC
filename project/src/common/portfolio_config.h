#ifndef PORTFOLIO_CONFIG_H
#define PORTFOLIO_CONFIG_H
#include "portfolio_common.h"
#include <cctype>
#include <map>
#include <regex>
#include <set>

// This file's contract is a flat JSON object containing the documented numeric fields.
// Reject unsupported structures/keys rather than silently ignoring a misspelled setting.
inline void load_run_config(const std::filesystem::path& path, SimConfig& c) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open config "+path.string());
    std::string text((std::istreambuf_iterator<char>(input)),{});
    if (text.compare(0,3,"\xef\xbb\xbf")==0) text.erase(0,3);
    size_t pos=0;
    auto spaces=[&]() { while(pos<text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos; };
    auto expect=[&](char ch) {
        spaces();
        if (pos==text.size() || text[pos++]!=ch) throw std::runtime_error("Malformed JSON config "+path.string());
    };
    expect('{'); spaces();
    std::set<std::string> seen;
    const std::regex number("-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?");
    if (pos<text.size() && text[pos]!='}') {
        while (true) {
            expect('"');
            const size_t begin=pos;
            while(pos<text.size() && text[pos]!='"') {
                if (text[pos]=='\\' || static_cast<unsigned char>(text[pos])<32)
                    throw std::runtime_error("Unsupported config key encoding");
                ++pos;
            }
            const std::string key=text.substr(begin,pos-begin);
            expect('"'); expect(':'); spaces();
            const size_t value_begin=pos;
            while(pos<text.size() && text[pos]!=',' && text[pos]!='}' &&
                  !std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
            const std::string value=text.substr(value_begin,pos-value_begin);
            if (!std::regex_match(value,number)) throw std::runtime_error("Expected a JSON number for "+key);
            if (!seen.insert(key).second) throw std::runtime_error("Duplicate config key: "+key);
            if (key=="K") c.K=parse_int(value,key);
            else if (key=="M") c.M=parse_integer(value,key);
            else if (key=="Rf") c.Rf=parse_float(value,key);
            else if (key=="R_min") c.R_min=parse_float(value,key);
            else if (key=="seed") c.seed=parse_seed(value);
            else if (key=="block_size") c.block_size=parse_int(value,key);
            else if (key=="cloud_sample_size") c.cloud_sample_size=parse_int(value,key);
            else if (key=="warmup_runs") c.warmup_runs=parse_int(value,key);
            else if (key=="repeats") c.repeats=parse_int(value,key);
            else throw std::runtime_error("Unknown config key: "+key);
            spaces();
            if (pos<text.size() && text[pos]==',') { ++pos; continue; }
            break;
        }
    }
    expect('}'); spaces();
    if (pos!=text.size()) throw std::runtime_error("Extra data after JSON config");
}

struct CommandLine {
    SimConfig config;
    std::filesystem::path root, output, config_path;
    std::string mode="all";
    bool pause=false, help=false;
};
inline std::filesystem::path find_project_root(const std::filesystem::path& executable) {
    for (const auto& start:{std::filesystem::current_path(),std::filesystem::absolute(executable).parent_path()}) {
        auto directory=start;
        for (int depth=0;depth<6;++depth) {
            for (const auto& candidate:{directory,directory/"project"})
                if (std::filesystem::is_regular_file(candidate/"data/processed/mu.txt"))
                    return std::filesystem::absolute(candidate).lexically_normal();
            if (directory==directory.parent_path()) break;
            directory=directory.parent_path();
        }
    }
    throw std::runtime_error("Cannot locate project data; pass --project-root <project>");
}
inline CommandLine parse_command_line(int argc,char** argv) {
    CommandLine options;
    std::map<std::string,std::string> args;
    const std::set<std::string> known={"--project-root","--config","--output-dir","--mode","--samples",
        "--block-size","--seed","--rf","--r-min","--cloud-size","--warmup","--repeats"};
    for (int i=1;i<argc;++i) {
        std::string key=argv[i];
        if (key=="--block_size") key="--block-size";
        if (key=="--help" || key=="-h") { options.help=true; continue; }
        if (key=="--pause") { options.pause=true; continue; }
        if (!known.count(key)) throw std::runtime_error("Unknown argument: "+key);
        if (i+1==argc) throw std::runtime_error("Missing value for "+key);
        if (!args.emplace(key,argv[++i]).second) throw std::runtime_error("Duplicate argument: "+key);
    }
    if (options.help) return options;
    options.root=args.count("--project-root")?std::filesystem::absolute(args.at("--project-root")).lexically_normal():
                                               find_project_root(argv[0]);
    options.config_path=args.count("--config")?std::filesystem::path(args.at("--config")):
                                               options.root/"config/run_config.json";
    if (options.config_path.is_relative()) options.config_path=options.root/options.config_path;
    if (args.count("--config") || std::filesystem::exists(options.config_path))
        load_run_config(options.config_path,options.config);
    auto& c=options.config;
    if (args.count("--mode")) options.mode=args.at("--mode");
    if (options.mode!="all" && options.mode!="verify" && options.mode!="mc")
        throw std::runtime_error("mode must be all, verify or mc");
    if (args.count("--samples")) c.M=parse_integer(args.at("--samples"),"M");
    if (args.count("--block-size")) c.block_size=parse_int(args.at("--block-size"),"block_size");
    if (args.count("--seed")) c.seed=parse_seed(args.at("--seed"));
    if (args.count("--rf")) c.Rf=parse_float(args.at("--rf"),"Rf");
    if (args.count("--r-min")) c.R_min=parse_float(args.at("--r-min"),"R_min");
    if (args.count("--cloud-size")) c.cloud_sample_size=parse_int(args.at("--cloud-size"),"cloud_sample_size");
    if (args.count("--warmup")) c.warmup_runs=parse_int(args.at("--warmup"),"warmup_runs");
    if (args.count("--repeats")) c.repeats=parse_int(args.at("--repeats"),"repeats");
    options.output=args.count("--output-dir")?std::filesystem::path(args.at("--output-dir")):options.root/"results/gpu";
    if (options.output.is_relative()) options.output=options.root/options.output;
    validate_config(c);
    return options;
}
#endif
